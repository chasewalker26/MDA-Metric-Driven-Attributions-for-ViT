import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from util.test_methods import MASTestFunctions as MAS_functions
from skimage.segmentation import slic
from skimage.util import img_as_float

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def normalize_curve(normalized_model_response, type = 0):
    n = len(normalized_model_response)
    Q = matrix(2 * np.eye(n))
    c = matrix(-2 * normalized_model_response)
    A_ineq = np.zeros((n - 2, n))
    b_ineq = np.full(n - 2, 0)
    row_indices = np.arange(n - 2)
    if type == 0:
        A_ineq[row_indices, row_indices] = -1
        A_ineq[row_indices, row_indices + 1] = 2
        A_ineq[row_indices, row_indices + 2] = -1
    elif type == 1:
        A_ineq[row_indices, row_indices] = 1
        A_ineq[row_indices, row_indices + 1] = -2
        A_ineq[row_indices, row_indices + 2] = 1
    G_bounds = np.vstack([-np.eye(n), np.eye(n)])
    h_bounds = np.hstack([np.zeros(n), np.ones(n)])
    G = matrix(np.vstack([G_bounds, A_ineq]))
    h = matrix(np.hstack([h_bounds, b_ineq]))
    A_eq = np.zeros((2, n))
    A_eq[0, 0] = 1
    A_eq[1, -1] = 1
    A_eq = matrix(A_eq)
    b_eq = matrix(np.array([normalized_model_response[0], normalized_model_response[-1]]), (2, 1), 'd')
    sol = solvers.qp(Q, c, G, h, A_eq, b_eq)
    return np.squeeze(np.array(sol['x']))

def find_insertion_patches(input_tensor, saliency_map_segmented, segments, blur, n_searches, type, model, device, img_hw, max_batch_size = 25, cutoff = 0.9):
    if cutoff == 0:
        return 0, 0, torch.tensor([]), torch.tensor([0])

    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    # deletion
    if type == 0:
        # set the start and stop images for each test
        start = torch.zeros_like(input_tensor)
        finish = input_tensor.clone()

        black_pred = model(start.to(device)).detach()
        percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
        black_pred = percentage[target_class].item()
    # insertion
    elif type == 1:
        # set the start and stop images for each test
        start = blur(input_tensor)
        finish = input_tensor.clone()

        blur_pred = model(start.to(device)).detach()
        percentage = torch.nn.functional.softmax(blur_pred, dim = 1)[0]
        blur_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    if type == 0:
        # segments ordered from lowest to highest attr value
        segment_order = torch.argsort(segment_saliency)
    elif type == 1:
        # segments ordered from highest to lowest attr value
        segment_order = torch.flip(torch.argsort(segment_saliency), dims=(0,))

    num_batches = int((n_steps) / batch_size)
    leftover = (n_steps) % batch_size
    batches = torch.full((1, num_batches + 1), batch_size).squeeze()
    batches[-1] = leftover
    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.empty((1, n_steps)).squeeze()

    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(n_searches - subsearch_len):
        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0

        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp
                total_steps += 1
            
            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        if type == 0:
            worst_MR_index = torch.argmin(model_response)
        elif type == 1:
            worst_MR_index = torch.argmax(model_response)

        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        if cutoff != 1 and ((worst_MR - blur_pred) / (abs(original_pred - blur_pred))) >= cutoff:
            worst_MR_list[step] = cutoff
            return 0, 0, worst_segment_list, worst_MR_list

    subsearch_len_orig = subsearch_len
    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step
        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        if type == 0:
            worst_MR_index = torch.argmin(model_response)
        elif type == 1:
            worst_MR_index = torch.argmax(model_response)

        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        if cutoff != 1 and ((worst_MR - blur_pred) / (abs(original_pred - blur_pred))) >= cutoff:
            worst_MR_list[step] = cutoff
            return 0, 0, worst_segment_list, worst_MR_list

    if type == 0:
        normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred])))
        normalized_model_response = torch.flip(normalized_model_response, (0,))
    elif type == 1:
        normalized_model_response = torch.cat((torch.tensor([blur_pred]), worst_MR_list))

    min_normalized_pred = 1.0
    max_normalized_pred = 0.0
    # perform monotonic normalization of raw model response
    normalized_model_response = normalized_model_response.detach().cpu().numpy().copy().astype(np.double)

    for i in range(n_steps + 1):           
        if type == 0:
            normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            min_normalized_pred = min(min_normalized_pred, normalized_pred)
            normalized_model_response[i] = min_normalized_pred
        elif type == 1:
            normalized_pred = (normalized_model_response[i] - blur_pred) / (abs(original_pred - blur_pred))
            normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
            max_normalized_pred = max(max_normalized_pred, normalized_pred)
            normalized_model_response[i] = max_normalized_pred

    original_MR = normalized_model_response.copy()
    normalized_model_response = normalize_curve(normalized_model_response, type)

    if type == 0:
        best_segment_list = torch.flip(worst_segment_list, (0,))
    elif type == 1:
        best_segment_list = worst_segment_list

    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        if type == 0:
            target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        elif type == 1:
            target_MR = normalized_model_response[i] - normalized_model_response[i - 1]

        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR

    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)
    upsize = transforms.Resize(img_hw)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side)

    return new_map.cpu().numpy(), upsize(downsize(new_map.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list, original_MR

def find_deletion_patches(input_tensor, segments, saliency_map_segmented, beginning_order, blur, n_searches, model, device, img_hw, max_batch_size = 25, kappa = 0.005, test_kappa = False):
    n_steps = len(np.unique(segments))
    
    batch_size = n_steps if n_steps < 50 else 25
    batch_size = max_batch_size if batch_size > max_batch_size else batch_size
    if batch_size > n_steps:
        print("Batch size cannot be greater than number of steps: " + str(n_steps))
        return 0

    # Retrieve softmax score of the original image
    original_pred = model(input_tensor.to(device)).detach()
    _, index = torch.max(original_pred, 1)
    target_class = index[0]
    percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
    original_pred = percentage[target_class].item()

    start = torch.zeros_like(input_tensor)
    
    finish = input_tensor.clone()
    black_pred = model(start.to(device)).detach()
    percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
    black_pred = percentage[target_class].item()

    saliency_map = torch.ones((224, 224, 3))
    saliency_map_test = torch.abs(torch.sum(saliency_map.squeeze(), axis = 2))

    saliency_map_segmented_test = torch.abs(torch.sum(saliency_map_segmented.squeeze(), axis = 2))
    segment_saliency = torch.zeros(n_steps)
    for i in range(n_steps):
        segment = torch.where(segments.flatten() == i)[0]
        segment_saliency[i] = torch.mean(saliency_map_segmented_test.reshape(img_hw ** 2)[segment])
    
    # We want to find the worst possible insertion order, so order segments from least to most attr value
    segment_order = torch.argsort(segment_saliency)

    worst_segment_list = torch.full((1, n_steps), -1).squeeze()
    worst_MR_list = torch.zeros((1, n_steps)).squeeze()

    # Set the last elements of the segment list to be the input from insertion since we know those are the best insertion values
    input_length = len(beginning_order)
    worst_segment_list[len(worst_segment_list) - input_length : len(worst_segment_list)] = torch.flip(beginning_order, (0,))
    
    subsearch_len = (int(n_steps ** (1/2)) * 2) if (int(n_steps ** (1/2)) * 2) <= 28 else 28
    for step in range(n_searches - subsearch_len - input_length):
        model_response = torch.zeros(subsearch_len)
        
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0
        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    if input_length > n_searches - subsearch_len:
        subsearch_len_orig = n_searches - input_length
    else:
        subsearch_len_orig = subsearch_len

    for step in range(subsearch_len_orig):
        subsearch_len = subsearch_len_orig - step

        model_response = torch.zeros(subsearch_len)
        # find the lowest subsearch_len segments that have not already been selected
        selected_segments = torch.zeros(subsearch_len)
        search_counter = 0
        index_counter = 0

        while index_counter != subsearch_len:
            if segment_order[search_counter] not in worst_segment_list:
                selected_segments[index_counter] = segment_order[search_counter]
                index_counter += 1
                search_counter += 1 
            else:
                search_counter += 1 

        batch_size = subsearch_len if subsearch_len < batch_size else batch_size 
        num_batches = int(subsearch_len / batch_size)
        leftover = subsearch_len % batch_size
        batches = torch.full((1, num_batches + 1), batch_size).squeeze()
        batches[-1] = leftover
        total_steps = 0
        for batch in batches:
            images = torch.zeros((batch.item(), input_tensor.shape[1], input_tensor.shape[2], input_tensor.shape[3]))
            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                start_temp = start.clone()
                segment_coords = torch.where(segments.flatten() == selected_segments[total_steps])[0]
                start_temp.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]
                images[i] = start_temp

                total_steps += 1

            # get predictions from image batch
            output = model(images.to(device)).detach()
            percentage = nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class]

        worst_MR_index = torch.argmin(model_response)
        worst_MR = model_response[worst_MR_index]
        worst_MR_list[step + n_searches - subsearch_len_orig - input_length] = worst_MR
        worst_segment = selected_segments[worst_MR_index]
        worst_segment_list[step + n_searches - subsearch_len_orig - input_length] = worst_segment

        segment_coords = torch.where(segments.flatten() == worst_segment)[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

    worst_segment_list_length = len(worst_segment_list)
    for step in range(worst_segment_list_length - input_length , worst_segment_list_length):
        segment_coords = torch.where(segments.flatten() == worst_segment_list[step])[0]
        start.reshape(3, img_hw ** 2)[:, segment_coords] = finish.reshape(3, img_hw ** 2)[:, segment_coords]

        # get predictions from image batch
        output = model(start.to(device)).detach()
        percentage = nn.functional.softmax(output, dim = 1)
        model_response = percentage[:, target_class]
        worst_MR_list[step] = model_response

    # this is the worst possible insertion curve
    normalized_model_response = torch.cat((worst_MR_list, torch.tensor([original_pred]))).detach().cpu().numpy().copy().astype(np.double)
    # flip it so it becomes the best possible deletion curve
    normalized_model_response = normalized_model_response[::-1]

    min_normalized_pred = 1.0
    # perform monotonic normalization of raw deletion model response
    for i in range(n_steps + 1):         
        normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
        normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
        min_normalized_pred = min(min_normalized_pred, normalized_pred)
        normalized_model_response[i] = min_normalized_pred

    # fix the derivative of the deletion curve
    normalized_model_response = normalize_curve(normalized_model_response, type = 0)

    # flip the worst ordered insertion list to become the best ordered deletion list 
    best_segment_list = torch.flip(worst_segment_list, (0,))
    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)
    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        new_map.reshape(img_hw ** 2)[segment_coords] = (ones_map.reshape(img_hw ** 2)[segment_coords] / len(segment_coords)) * target_MR + (target_MR * (n_steps - i) / n_steps)
    new_map = new_map.unsqueeze(2) * torch.ones_like(saliency_map)

    # get insertion and deletion curves for the current attribution
    MAS_insertion = MAS_functions.MASMetric(model, img_hw ** 2, 'ins', img_hw, substrate_fn = blur)
    MAS_deletion = MAS_functions.MASMetric(model, img_hw ** 2, 'del', img_hw, substrate_fn = torch.zeros_like)
    new_map_test = np.abs(np.sum(new_map.cpu().numpy(), axis = 2))
    _, _, _, _, raw_score_ins = MAS_insertion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    _, _, _, _, raw_score_del = MAS_deletion.single_run(input_tensor, new_map_test, device, max_batch_size=5)
    # interpolate the two curves to the correct number of steps
    x_old = np.linspace(0, 100, len(raw_score_ins))
    x_new = np.linspace(0, 100, n_steps + 1)
    raw_score_ins = np.interp(x_new, x_old, raw_score_ins)
    raw_score_del = np.interp(x_new, x_old, raw_score_del)
    # take the mean of the two curves, treat it as a deletion curve
    new_curve = np.mean([raw_score_ins, 1 - raw_score_del], axis = 0)
    new_curve = 1 - new_curve
    # fix derivative, this is the new model response
    normalized_model_response = normalize_curve(new_curve, type = 0)

    # using the model response, set each patch to be the correct value for perfect order deletion
    new_map_sparse = torch.zeros_like(saliency_map_test)
    new_map_dense = torch.zeros_like(saliency_map_test)
    ones_map = torch.ones_like(saliency_map_test)

    if test_kappa:
        return new_map_dense, segments, best_segment_list, normalized_model_response, ones_map, n_steps + 1

    for i in range(1, n_steps + 1):
        segment_coords = torch.where(segments.flatten() == best_segment_list[i-1])[0]
        target_MR = normalized_model_response[i - 1] - normalized_model_response[i]
        attr_value = 1 / len(segment_coords) * target_MR + (target_MR * (n_steps - i) / n_steps) 

        new_map_sparse.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value
        if attr_value >= kappa:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * ((n_steps - i) / n_steps)
        else:
            new_map_dense.reshape(img_hw ** 2)[segment_coords] = ones_map.reshape(img_hw ** 2)[segment_coords] * attr_value
    
    # put the dense and sparse maps on the same range
    new_map_dense = new_map_dense / new_map_dense.max() * new_map_sparse.max()

    new_map_0 = ((1 - 0) * new_map_sparse) + ((0) * new_map_dense)
    new_map_5 = ((1 - 0.5) * new_map_sparse) + ((0.5) * new_map_dense)
    new_map_10 = ((1 - 1) * new_map_sparse) + ((1) * new_map_dense)
    new_map_0 = new_map_0.unsqueeze(2) * torch.ones_like(saliency_map)
    new_map_5 = new_map_5.unsqueeze(2) * torch.ones_like(saliency_map)
    new_map_10 = new_map_10.unsqueeze(2) * torch.ones_like(saliency_map)

    upsize = transforms.Resize(img_hw)
    small_side = int(np.ceil(np.sqrt(n_steps)))
    downsize = transforms.Resize(small_side)

    return new_map_0.cpu().numpy(), upsize(downsize(new_map_0.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), new_map_5.cpu().numpy(), upsize(downsize(new_map_5.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), new_map_10.cpu().numpy(), upsize(downsize(new_map_10.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy(), best_segment_list


def MDA(trans_img, input_tensor, saliency_map, patch_count, blur, model, device, img_hw, max_batch_size = 5):
    # draw patches on the input image
    segment_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))
    segment_img = img_as_float(segment_img)
    segments = torch.tensor(slic(segment_img, n_segments=patch_count, compactness=10000, start_label=0), dtype = int)

    # convert the input attribution to a patch x patch attribution
    upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT)
    downsize = transforms.Resize(int(patch_count ** (1/2)))
    saliency_map_segmented = upsize(downsize(torch.tensor(saliency_map, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))

    # find the patches from insertion and transfer them to deletion to find all patches
    _, _, order_a, MR_ins = find_insertion_patches(input_tensor.cpu(), saliency_map_segmented, segments, blur, patch_count, type = 1, model = model, device = device, img_hw = img_hw, max_batch_size = 5)
    end_index = np.where(MR_ins >= 0.9)[0][0]
    _, MGA_g_0, _, MGA_g_5,  _, MGA_g_10,  _ = find_deletion_patches(input_tensor.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, patch_count, model, device, img_hw, max_batch_size = 5)

    return MGA_g_0, MGA_g_5, MGA_g_10