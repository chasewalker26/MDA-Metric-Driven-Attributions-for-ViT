import torch
import torch.nn as nn
from torchvision import transforms
import csv
import argparse
import time
import numpy as np
from PIL import Image
import os
import warnings
from skimage.segmentation import slic
from skimage.util import img_as_float
os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.test_methods import MASTestFunctions as MAS
from util.test_methods import MonotonicityTest as MT
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods import MDAFunctions
from util.attribution_methods.TIS import TIS
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX

# models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_tiny_patch16_224 as vit_new_tiny_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_tiny_patch16_224 as vit_LRP_tiny_16
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224 as vit_new_base_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_LRP_base_16
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224 as vit_new_base_32
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_LRP_base_32
model = None

# standard ImageNet normalization
transform_normalize_VIT = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

# standard ImageNet normalization
transform_normalize_RES = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


resize = transforms.Resize((224, 224), antialias = True)

# runs an attribution method w 3 baselines over imageCount images and calculates the mean PIC
def run_and_save_tests(img_hw, image_count, function, transform, model, explainer, LRP_explainer, model_name, device, imagenet, batch_size, cutoff = 0.9, kappa = 0.5):
    # num imgs used for testing
    img_label = str(image_count) + "_images_"

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    num_classes = 1000
    images_per_class = int(np.ceil(image_count / num_classes))
    classes_used = [0] * num_classes

    fields = ["attr", "Pos Monotonicity", "Neg Monotonicity"]
    scores = [function, 0, 0]

    images = sorted(os.listdir(imagenet))
    images_used = 0

    if model_name == "R101":
        segment_count = 14 ** 2
    elif model_name == "VIT_tiny_16":
        segment_count = 14 ** 2
    elif model_name == "VIT_base_16":
        segment_count = 14 ** 2
    elif model_name == "VIT_base_32":
        segment_count = 7 ** 2
 
    total_time = 0

    # look at test images in order from 1
    for image in images:    
        if images_used == image_count:
            print("method finished")
            break

        begin = time.time()

        # check if the current image is an invalid image for testing, 0 indexed
        image_num = int((image.split("_")[2]).split(".")[0]) - 1
        # check if the current image is an invalid image for testing
        if correctly_classified[image_num] == 0:
            continue

        img = Image.open(imagenet + "/" + image)
        trans_img = transform(img)

        # put the image in form needed for prediction for the ins/del method
        if model_name == "VIT_tiny_16":
            img_tensor = transform_normalize_VIT(trans_img)
        elif model_name == "VIT_base_16":
            img_tensor = transform_normalize_VIT(trans_img)
        elif model_name == "VIT_base_32":
            img_tensor = transform_normalize_VIT(trans_img)
        elif model_name == "R101":
            img_tensor = transform_normalize_RES(trans_img)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(device)

        # only rgb images can be classified
        if trans_img.shape != (3, img_hw, img_hw):
            continue

        target_class = model_utils.getClass(img_tensor, model, device)
        original_pred = model_utils.getPrediction(img_tensor, model, device, target_class)[0] * 100
        if original_pred < 60:
            continue

        # Track which classes have been used
        if classes_used[target_class] == images_per_class:
            continue
        else:
            classes_used[target_class] += 1       

        klen = 1
        ksig = 1
        kern = MAS.gkern(klen, ksig)
        blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
        blur_pred = model_utils.getPrediction(blur(img_tensor.cpu()).to(device), model, device, target_class)[0] * 100
        # choose a blurring kernel that results in a softmax score of 1% or less 
        while blur_pred > 1 and klen <= 101:
            klen += 2
            ksig += 2
            kern = MAS.gkern(klen, ksig)
            blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
            blur_pred = model_utils.getPrediction(blur(img_tensor.cpu()).to(device), model, device, target_class)[0] * 100

        if klen > 101:
          classes_used[target_class] -= 1
          continue

        MT_class_pos = MT.MonotonicityMetric(model, img_hw * img_hw, 'positive', img_hw, substrate_fn = blur)
        MT_class_neg = MT.MonotonicityMetric(model, img_hw * img_hw, 'negative', img_hw, substrate_fn = blur)

        print(model_name + " Function " + function + ", image: " + image)

        if (function == 'IG' and model_name != "R101"):    
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            saliency_map = IG.permute(1, 2, 0).numpy()
        elif (function == 'GC' and model_name != "R101"):    
            attr = explainer.generate_cam_attn(img_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = attr.permute(1, 2, 0).numpy()
        elif (function == 'LRP' and model_name != "R101"):
            attr = LRP_explainer.generate_LRP(img_tensor, target_class, method="transformer_attribution", start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0).numpy()
        elif (function == 'Transition_attn' and model_name != "R101"):
            _, _, attr, _, _ = explainer.generate_transition_attention_maps(img_tensor, target_class, start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0).numpy()
        elif (function == 'Bidirectional' and model_name != "R101"):
            attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            saliency_map = (attr).permute(1, 2, 0).numpy()
        elif (function == "TIS" and model_name != "R101"):
            saliency_method = TIS(model, batch_size=64)
            saliency_map = saliency_method(img_tensor, class_idx=target_class).cpu()
            saliency_map = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
        elif (function == "VIT_CX" and model_name != "R101"):
            target_layer = model.blocks[-1].norm1
            result, features = ViT_CX(model, img_tensor, target_layer, gpu_batch=1, device = device)
            saliency_map = (result.reshape((224, 224, 1)) * torch.ones((224, 224, 3))).cpu().detach().numpy()
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        elif (function == 'Calibrate'):
            start_2 = time.time()
            segment_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))
            segment_img = img_as_float(segment_img)
            segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
            attr_attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
            reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
            upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT)
            downsize = transforms.Resize(int(segment_count ** (1/2)))
            saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(img_tensor.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = img_hw, max_batch_size = batch_size)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            _, saliency_map = MDAFunctions.find_deletion_patches(img_tensor.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, img_hw, max_batch_size = batch_size, kappa = -1)
            saliency_map = saliency_map.detach().cpu().numpy()
            total_time += time.time() - start_2
        elif (function == 'Calibrate_Ins'):
            segment_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))
            segment_img = img_as_float(segment_img)
            segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
            attr_attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
            reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
            upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT)
            downsize = transforms.Resize(int(segment_count ** (1/2)))
            saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
            _, saliency_map, _, _ = MDAFunctions.find_insertion_patches(img_tensor.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = img_hw, max_batch_size = batch_size, cutoff = 1)
        elif (function == 'Calibrate_Del'):
            segment_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))
            segment_img = img_as_float(segment_img)
            segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
            attr_attr, _ = explainer.bidirectional(img_tensor, target_class, device = device)
            reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
            upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT)
            downsize = transforms.Resize(int(segment_count ** (1/2)))
            saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
            _, saliency_map = MDAFunctions.find_deletion_patches(img_tensor.cpu(), segments, saliency_map_segmented, torch.tensor([]), blur, segment_count, model, device, img_hw, max_batch_size = batch_size)
            upsize = transforms.Resize(224, antialias=True)
            small_side = int(np.ceil(np.sqrt(segment_count)))
            downsize = transforms.Resize(small_side, antialias=True)
            saliency_map = upsize(downsize(saliency_map.permute((2, 0, 1)))).permute((1, 2, 0)).cpu().numpy()
        else:
            print("You have not picked a valid attribution method.")

        # Get attribution scores
        saliency_map_test = np.abs(np.sum(saliency_map, axis = 2, keepdims=True))

        # make sure attribution is valid
        if np.sum(saliency_map_test.reshape(1, 1, img_hw ** 2)) == 0:
            print("Skipping Image due to 0 attribution")
            classes_used[target_class] -= 1
            continue


        ins_del_img = img_tensor.cpu()

        scores[1] += MT_class_pos.single_run(ins_del_img, saliency_map_test, device, max_batch_size = batch_size)
        scores[2] += MT_class_neg.single_run(ins_del_img, saliency_map_test, device, max_batch_size = batch_size)

        # when all tests have passed, the number of images used can go up by 1
        images_used += 1

        print("Total used: " + str(images_used) + " / " + str(image_count))

        print(time.time() - begin)

    print("Time Elapsed = " + str(total_time / images_used))

    for i in range(1, 3):
        scores[i] /= images_used
        scores[i] = round(scores[i], 3)

    # make the test folder if it doesn't exist
    folder = "test_results_init/imagenet/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    if function == "Calibrate_Cutoff":
        img_label = model_name + "_" + function + "_" + str(cutoff) + "_" + str(image_count) + "_images"
    elif function == "Calibrate_Kappa":
        img_label = model_name + "_" + function + "_" + str(kappa) + "_" + str(image_count) + "_images"
    else:
        img_label = model_name + "_" + function + "_" + str(image_count) + "_images"
    with open(folder + "Monotonicity_" + img_label + ".csv", 'w') as f:
        write = csv.writer(f)
        write.writerow(fields)
        write.writerow(scores)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.gpu) if torch.cuda.is_available() else 'cpu'

    model_name = FLAGS.model

    batch_size = 25

    if model_name == "VIT_tiny_16":
        model = vit_new_tiny_16(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_tiny_16(pretrained=True).to(device).eval()
        explainer = Baselines(model)
        LRP_explainer = LRP(model_lrp)
    elif model_name == "VIT_base_16":
        model = vit_new_base_16(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_base_16(pretrained=True).to(device).eval()
        explainer = Baselines(model)
        LRP_explainer = LRP(model_lrp)
    elif model_name == "VIT_base_32":
        model = vit_new_base_32(pretrained=True).to(device).eval()
        model_lrp = vit_LRP_base_32(pretrained=True).to(device).eval()
        explainer = Baselines(model)
        LRP_explainer = LRP(model_lrp)

    img_hw = 224
    transform = transforms.Compose([
        transforms.Resize(img_hw),
        transforms.CenterCrop(img_hw),
        transforms.ToTensor()
    ])

    run_and_save_tests(img_hw, FLAGS.image_count, FLAGS.function, transform, model, explainer, LRP_explainer, model_name, device, FLAGS.imagenet, batch_size, FLAGS.cutoff, FLAGS.kappa)

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Attribution Test Script.')
    parser.add_argument('--function',
                        type = str, default = "IG",
                        help = 'Name of the attribution method to use: .')
    parser.add_argument('--model',
                        type = str, default = "VIT_base_16",
                        help = 'Name of the model to use: VIT_tiny_16, VIT_base_32, VIT_base_16.')
    parser.add_argument('--cutoff',
                    type = float, default = 0.9,
                    help = '[0, 1.0]')
    parser.add_argument('--kappa',
                    type = float, default = 0.5,
                    help = '[0, 100]')
    parser.add_argument('--image_count',
                        type = int, default = 5000,
                        help='How many images to test with.')
    parser.add_argument('--gpu',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet validation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)