# this code is from https://github.com/chasewalker26/Magnitude-Aligned-Scoring
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

# this code borrows certain functions from 
# https://github.com/eclique/RISE/blob/master/evaluation.py

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""

    # create nxn zeros
    inp = np.zeros((klen, klen))

    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1

    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k

    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class MASMetric():
    def __init__(self, model, HW, mode, step_size, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            HW (int): image size in pixels given as h*w e.g. 224*224.
            mode (str): 'del' or 'ins'.
            step_size (int): number of pixels modified per one iteration e.g. 224.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.HW = HW
        self.mode = mode
        self.step_size = step_size
        self.substrate_fn = substrate_fn

    def single_run(self, img_tensor, saliency_map, device, max_batch_size = 50):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            saliency_map (np.ndarray): saliency map.
            device (str): 'cpu' or gpu id e.g. 'cuda:0'.
            max_batch_size (int): controls the parallelization of the test.
        Return:
            n_steps (int): the number of steps used over the test.
            corrected_scores (nd.array): Array containing MAS scores at every step.
            alignment_penalty (nd.array): Array containing alignment penalty at every step.
            density_response (nd.array): Array containing the density response at every step.
            normalized_model_response (nd.array): Array containing the model response at every step.
        """

        n_steps = (self.HW + self.step_size - 1) // self.step_size

        batch_size = n_steps if n_steps < max_batch_size else max_batch_size

        if batch_size > n_steps:
            print("Batch size cannot be greater than number of steps: " + str(n_steps))
            return 0, 0, 0, 0, 0

        # Retrieve softmax score of the original image
        original_pred = self.model(img_tensor.to(device)).detach()
        _, index = torch.max(original_pred, 1)
        target_class = index[0]
        percentage = torch.nn.functional.softmax(original_pred, dim = 1)[0]
        original_pred = percentage[target_class].item()

        model_response = np.zeros(n_steps + 1)

        # set the start and stop images for each test
        # get softmax score of the subtrate-applied images
        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)

            black_pred = self.model(finish.to(device)).detach()
            percentage = torch.nn.functional.softmax(black_pred, dim = 1)[0]
            black_pred = percentage[target_class].item()

            model_response[0] = original_pred
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

            blur_pred = self.model(start.to(device)).detach()
            percentage = torch.nn.functional.softmax(blur_pred, dim = 1)[0]
            blur_pred = percentage[target_class].item()

            model_response[0] = blur_pred

        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(saliency_map.reshape(-1, self.HW), axis = 1), axis = -1)

        density_response = np.zeros(n_steps + 1)

        if self.mode == "del":
            density_response[0] = 1
        elif self.mode == "ins":
            density_response[0] = 0

        total_attr = np.sum(saliency_map.reshape(1, 1, self.HW))

        total_steps = 1
        num_batches = int((n_steps) / batch_size)
        leftover = (n_steps) % batch_size

        batches = np.full(num_batches + 1, batch_size)
        batches[-1] = leftover

        for batch in batches:
            images = torch.zeros((batch, start.shape[1], start.shape[2], start.shape[3]))

            # collect images at all batch steps before mass prediction 
            for i in range(batch):
                coords = salient_order[:, self.step_size * (total_steps - 1) : self.step_size * (total_steps)]
                start.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords] = finish.cpu().numpy().reshape(1, 3, self.HW)[0, :, coords]
                images[i] = start

                attr_count = np.sum(saliency_map.reshape(1, 1, self.HW)[0, :, coords]) 
                if self.mode == "del":
                    density_response[total_steps] = density_response[total_steps - 1] - (attr_count / total_attr)
                elif self.mode == "ins":
                    density_response[total_steps] = density_response[total_steps - 1] + (attr_count / total_attr)

                total_steps += 1

            # get predictions from image batch
            output = self.model(images.to(device)).detach()
            percentage = torch.nn.functional.softmax(output, dim = 1)
            model_response[total_steps - batch : total_steps] = percentage[:, target_class].cpu().numpy()

        min_normalized_pred = 1.0
        max_normalized_pred = 0.0
        # perform monotonic normalization of raw model response
        normalized_model_response = model_response.copy()
        for i in range(n_steps + 1):           
            if self.mode == 'del':
                normalized_pred = (normalized_model_response[i] - black_pred) / (abs(original_pred - black_pred))
                normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
                min_normalized_pred = min(min_normalized_pred, normalized_pred)
                normalized_model_response[i] = min_normalized_pred
            elif self.mode == 'ins':
                normalized_pred = (normalized_model_response[i] - blur_pred) / (abs(original_pred - blur_pred))
                normalized_pred = np.clip(normalized_pred, 0.0, 1.0)
                max_normalized_pred = max(max_normalized_pred, normalized_pred)
                normalized_model_response[i] = max_normalized_pred

        alignment_penalty = np.abs(normalized_model_response - density_response)

        if self.mode == "del":
            corrected_scores = normalized_model_response + alignment_penalty    
        elif self.mode == "ins":
            corrected_scores = normalized_model_response - alignment_penalty

        # scores should be clipped before normalization or else values outside of these bounds will artificially improve the final score
        corrected_scores = corrected_scores.clip(0, 1)
        corrected_scores = (corrected_scores - np.min(corrected_scores)) / (np.max(corrected_scores) - np.min(corrected_scores))

        # if somehow the blurred or black image recieved the same prediction as the input image, causing a failure of the above line, assign the attribution an ROC with a score of 0.5
        if np.isnan(corrected_scores).any():
            if self.mode == 'del':
                corrected_scores = np.linspace(1, 0, n_steps + 1)
            elif self.mode == 'ins':
                corrected_scores = np.linspace(0, 1, n_steps + 1)

        return n_steps + 1, corrected_scores, alignment_penalty, density_response, normalized_model_response