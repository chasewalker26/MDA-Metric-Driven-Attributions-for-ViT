import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from numpy import *
import argparse
from PIL import Image
import imageio
import os
import warnings
import time
from tqdm import tqdm
from utils.metrices import *
from torch import nn

from utils import render
from utils.saver import Saver
from utils.iou import IoU
from skimage.segmentation import slic
from skimage.util import img_as_float
from data.Imagenet import Imagenet_Segmentation
from captum.attr import ShapleyValueSampling

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.visualization import attr_to_subplot
from util.attribution_methods import MDAFunctions
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods.TIS import TIS
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX
from util.test_methods import MASTestFunctions as MAS

# models
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_base_patch16_224_LRP

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_base_patch32_224_LRP


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

import torch.nn.functional as F

# plt.switch_backend('agg')

# heatmap
def heatmap_overlap_axs(img, attr, axs, name):
    attr_to_subplot(attr, name, axs, cmap = 'jet', norm = 'absolute', blended_image=img, alpha = 0.6)
    return

# RUN WITH
# python3 baselines/ViT/imagenet_seg_eval.py --method transformer_attribution --imagenet-seg-path ../../../gtsegs_ijcv.mat

# hyperparameters
num_workers = 0
batch_size = 1

cls = ['airplane',
       'bicycle',
       'bird',
       'boat',
       'bottle',
       'bus',
       'car',
       'cat',
       'chair',
       'cow',
       'dining table',
       'dog',
       'horse',
       'motobike',
       'person',
       'potted plant',
       'sheep',
       'sofa',
       'train',
       'tv'
       ]

# Args
parser = argparse.ArgumentParser(description='Training multi-class classifier')
parser.add_argument('--arc', type=str, default='vgg', metavar='N',
                    help='Model architecture')
parser.add_argument('--train_dataset', type=str, default='imagenet', metavar='N',
                    help='Testing Dataset')
parser.add_argument('--method', type=str,
                    default='grad_rollout',
                    help='')
parser.add_argument('--model', type=str,
                    default='VIT_base_16',
                    help='VIT_base_16 or VIT_base_32')
parser.add_argument('--thr', type=float, default=0.,
                    help='threshold')
parser.add_argument('--threshold', type=float, default=0.6,
                    help='threshold for acc, IoU, mAP...')
parser.add_argument('--kappa', type=float, default=0.6,
                    help='kapa...')
parser.add_argument('--acc_cutoff', type=float, default=60,
                    help='')
parser.add_argument('--K', type=int, default=1,
                    help='new - top K results')
parser.add_argument('--save-img', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-ia', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-fgx', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-m', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--no-reg', action='store_true',
                    default=False,
                    help='')
parser.add_argument('--is-ablation', type=bool,
                    default=False,
                    help='')
parser.add_argument('--imagenet-seg-path', type=str, required=True)
parser.add_argument('--gpu', type=int, default = 0,
                    help='The number of the GPU you want to use.')
args = parser.parse_args()

args.checkname = args.model + '/' + args.method + '_' + args.arc

alpha = 2

cuda = torch.cuda.is_available()
device = 'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'

# Define Saver
saver = Saver(args)
saver.results_dir = os.path.join(saver.experiment_dir, '../test_results')

print(saver.results_dir)

if not os.path.exists(saver.results_dir):
    os.makedirs(saver.results_dir)
if not os.path.exists(os.path.join(saver.results_dir, 'input')):
    os.makedirs(os.path.join(saver.results_dir, 'input'))
if not os.path.exists(os.path.join(saver.results_dir, 'explain')):
    os.makedirs(os.path.join(saver.results_dir, 'explain'))

args.exp_img_path = os.path.join(saver.results_dir, 'explain/img')
if not os.path.exists(args.exp_img_path):
    os.makedirs(args.exp_img_path)
args.exp_np_path = os.path.join(saver.results_dir, 'explain/np')
if not os.path.exists(args.exp_np_path):
    os.makedirs(args.exp_np_path)

# Data
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

# invert standard ImageNet normalization
inv_normalize = transforms.Normalize(
    mean=[-1, -1, -1],
    std=[2, 2, 2]
)


test_img_trans = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])
test_lbl_trans = transforms.Compose([
    transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT),
])

resize = transforms.Resize((224, 224), antialias = True)


ds = Imagenet_Segmentation(args.imagenet_seg_path,
                           transform=test_img_trans, target_transform=test_lbl_trans)
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)


if args.model == "VIT_base_16":
    model = vit_base_patch16_224(pretrained=True).to(device)
    model_LRP = vit_base_patch16_224_LRP(pretrained=True).to(device)
    segment_count = 14 ** 2 
elif args.model == "VIT_base_32":
    model = vit_base_patch32_224(pretrained=True).to(device)
    model_LRP = vit_base_patch32_224_LRP(pretrained=True).to(device)
    segment_count = 7 ** 2 


baselines = Baselines(model)
model_LRP.eval()
lrp = LRP(model_LRP)

metric = IoU(2, ignore_index=-1)

iterator = tqdm(dl)

model.eval()


def compute_pred(output):
    pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
    T = pred.squeeze().cpu().numpy()
    T = np.expand_dims(T, 0)
    T = (T[:, np.newaxis] == np.arange(1000)) * 1.0
    T = torch.from_numpy(T).type(torch.FloatTensor)
    Tt = T.to(device)

    return Tt


def eval_batch(image, labels, evaluator, index, acc_cutoff, threshold, kappa):
    evaluator.zero_grad()
    # Save input image
    if args.save_img:
        img = image[0].permute(1, 2, 0).data.cpu().numpy()
        img = 255 * (img - img.min()) / (img.max() - img.min())
        img = img.astype('uint8')
        Image.fromarray(img, 'RGB').save(os.path.join(saver.results_dir, 'input/{}_input.png'.format(index)))
        Image.fromarray((labels.repeat(3, 1, 1).permute(1, 2, 0).data.cpu().numpy() * 255).astype('uint8'), 'RGB').save(
            os.path.join(saver.results_dir, 'input/{}_mask.png'.format(index)))

    target_class = model_utils.getClass(image, model, device)

    original_pred = model_utils.getPrediction(image, model, device, target_class)[0] * 100
    if original_pred < acc_cutoff:
        return 0, 0, 0, 0, 0, 0, 0, 0

    klen = 1
    ksig = 1
    kern = MAS.gkern(klen, ksig)
    blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
    blur_pred = model_utils.getPrediction(blur(image.cpu()).to(device), model, device, target_class)[0] * 100
    # choose a blurring kernel that results in a softmax score of 1% or less 
    while blur_pred > 1 and klen <= 101:
        klen += 2
        ksig += 2
        kern = MAS.gkern(klen, ksig)
        blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
        blur_pred = model_utils.getPrediction(blur(image.cpu()).to(device), model, device, target_class)[0] * 100

    if klen > 101:
        return 0, 0, 0, 0, 0, 0, 0, 0

    image.requires_grad = True

    image = image.requires_grad_()
    predictions = evaluator(image)
    
    if (args.method == 'IG'):    
        _, IG, _, _, _ = baselines.generate_transition_attention_maps(image, target_class, start_layer = 0, device = device)
        IG = resize(IG.cpu().detach())
        Res = (IG).permute(1, 2, 0)
        Res = Res.reshape(batch_size, 1, 224, 224)

    elif (args.method == 'GC'):    
        attr = baselines.generate_cam_attn(image, target_class, device = device)
        attr = resize(attr.cpu().detach())
        Res = (attr).permute(1, 2, 0)
        Res = Res.reshape(batch_size, 1, 224, 224)

    elif (args.method == 'LRP'):
        attr = lrp.generate_LRP(image, target_class, method="transformer_attribution", start_layer = 0, device = device)
        attr = resize(attr.cpu().detach())
        Res = (attr).permute(1, 2, 0)
        Res = Res.reshape(batch_size, 1, 224, 224)

    elif (args.method == 'Transition_attn'):
        _, _, attr, _, _ = baselines.generate_transition_attention_maps(image, target_class, start_layer = 0, device = device)
        attr = resize(attr.cpu().detach())
        Res = (attr).permute(1, 2, 0)
        Res = Res.reshape(batch_size, 1, 224, 224)

    elif (args.method == 'Bidirectional'):
        attr, _ = baselines.bidirectional(image, target_class, device = device)
        attr = resize(attr.cpu().detach())
        Res = (attr).permute(1, 2, 0)
        Res = Res.reshape(batch_size, 1, 224, 224)

    elif (args.method == "TIS"):
        saliency_method = TIS(model, batch_size=64)
        saliency_map = saliency_method(image, class_idx=target_class).cpu()
        Res = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 1))
        Res = Res.reshape(batch_size, 1, 224, 224)
        Res = torch.from_numpy(Res)

    elif (args.method == "VIT_CX"):
        target_layer = model.blocks[-1].norm1
        result, _ = ViT_CX(model, image, target_layer, gpu_batch=1, device = device)
        saliency_map = (result.reshape((224, 224, 1)) * torch.ones((224, 224, 1))).cpu().detach().numpy()
        Res = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
        Res = Res.reshape(batch_size, 1, 224, 224)
        Res = torch.from_numpy(Res)

    elif (args.method == "SHAP"):
        trans_img = inv_normalize(image.squeeze())
        segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
        segment_img = img_as_float(segment_img)
        segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
        svs = ShapleyValueSampling(model)
        attr = svs.attribute(image, target=target_class, feature_mask=segments.reshape(1, 1, 224, 224).cuda()).squeeze().permute((1, 2, 0))
        upsize = transforms.Resize(224, antialias=True)
        small_side = int(np.ceil(np.sqrt(segment_count)))
        downsize = transforms.Resize(small_side, antialias=True)
        Res = upsize(downsize(attr.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(batch_size, 1, 224, 224)

    elif 'Calibrate' in args.method:
        trans_img = inv_normalize(image.squeeze())
        segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
        segment_img = img_as_float(segment_img)
        segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
        attr_attr, _ = baselines.bidirectional(image, target_class, device = device)
        reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
        upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias = True)
        downsize = transforms.Resize(int(segment_count ** (1/2)), antialias = True)
        saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))


        if (args.method == 'Calibrate_Sparse'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            Res, _ = MDAFunctions.find_deletion_patches(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25)
            Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)

        elif (args.method == 'Calibrate_Dense'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            _, Res = MDAFunctions.find_deletion_patches(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25)
            Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)

        elif (args.method == 'Calibrate_Sparse_Smooth'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            Res, _ = MDAFunctions.find_deletion_patches(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = batch_size)
            
            upsize = transforms.Resize(224, antialias=True)
            small_side = int(np.ceil(np.sqrt(segment_count)))
            downsize = transforms.Resize(small_side, antialias=True)
            Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(batch_size, 1, 224, 224)

        elif (args.method == 'Calibrate_Dense_Smooth'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            Res, _ = MDAFunctions.find_deletion_patches(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25)

            upsize = transforms.Resize(224, antialias=True)
            small_side = int(np.ceil(np.sqrt(segment_count)))
            downsize = transforms.Resize(small_side, antialias=True)
            Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(batch_size, 1, 224, 224)

        elif (args.method == 'Calibrate_Dense_Smooth_a'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            new_map_dense_orig, segments, best_segment_list, normalized_model_response, ones_map, n_steps = MDAFunctions.find_deletion_from_insertion_informed_ultimate(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = batch_size, test_kappa = True)
            
            # testing various kappa values
            new_map_dense = new_map_dense_orig.clone()
            for j in range(1, n_steps):
                segment_coords = torch.where(segments.flatten() == best_segment_list[j-1])[0]
                target_MR = normalized_model_response[j - 1] - normalized_model_response[j]
                attr_value = 1 / len(segment_coords) * target_MR + (target_MR * (n_steps - j) / n_steps) 

                if attr_value >= 0.005:
                    new_map_dense.reshape(224 ** 2)[segment_coords] = ones_map.reshape(224 ** 2)[segment_coords] * 1
                else:
                    new_map_dense.reshape(224 ** 2)[segment_coords] = ones_map.reshape(224 ** 2)[segment_coords] * 0

            Res = new_map_dense.unsqueeze(2) * torch.ones((224, 224, 3))
            upsize = transforms.Resize(224, antialias=True)
            small_side = int(np.ceil(np.sqrt(segment_count)))
            downsize = transforms.Resize(small_side, antialias=True)
            Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(batch_size, 1, 224, 224)

        elif (args.method == 'Calibrate_Dense_Smooth_b') or (args.method == 'Calibrate_Dense_Varied') or (args.method == 'Calibrate_Best_Possible'):
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(image.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            _, Res = MDAFunctions.find_deletion_patches(image.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = batch_size, kappa = -1)
            
            upsize = transforms.Resize(224, antialias=True)
            small_side = int(np.ceil(np.sqrt(segment_count)))
            downsize = transforms.Resize(small_side, antialias=True)
            Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0)).mean(dim=-1).reshape(batch_size, 1, 224, 224)


    if 'Calibrate_Dense_Smooth_a' in args.method: 
        ret = (Res.max() - Res.min()) / 2
    elif 'Calibrate_Dense_Smooth_b' in args.method: 
        ret = Res.mean()
    elif 'Calibrate_Dense_Varied' in args.method:
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        ret = threshold
    elif 'Calibrate_Best_Possible' in args.method:
        # evaluate all possible thresholds to pick the best performing
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        # vary kappa
        mag_vals = np.array([0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])

        Res = Res / Res.mean() * 0.5

        acc_array = np.zeros(len(mag_vals))
        IoU_array = np.zeros(len(mag_vals))
        AP_array = np.zeros(len(mag_vals))
        f1_array = np.zeros(len(mag_vals))
        for i in range(len(mag_vals)):
            Res_mean_1 = Res.gt(mag_vals[i]).type(Res.type())
            Res_mean_0 = Res.le(mag_vals[i]).type(Res.type())
            output = torch.cat((Res_mean_0, Res_mean_1), 1)
            correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
            inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)

            AP_array[i] = np.nan_to_num(get_ap_scores(output, labels))[0]
            acc_array[i] = np.float64(1.0) * correct / (np.spacing(1, dtype=np.float64) + labeled).squeeze()
            IoU_array[i] = np.mean(np.float64(1.0) * inter / (np.spacing(1, dtype=np.float64) + union))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f1_array[i] = np.mean(np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0])))

        max_AP = np.argmax(AP_array)
        max_IoU = np.argmax(IoU_array)
        max_f1 = np.argmax(f1_array)

        selected_index = np.argmax(IoU_array)

        ret = mag_vals[selected_index]
    else:
        # threshold between FG and BG is the mean    
        Res = (Res - Res.min()) / (Res.max() - Res.min())
        ret = Res.mean()

    Res_1 = Res.gt(ret).type(Res.type())
    Res_0 = Res.le(ret).type(Res.type())

    Res_1_AP = Res
    Res_0_AP = 1-Res

    Res_1[Res_1 != Res_1] = 0
    Res_0[Res_0 != Res_0] = 0
    Res_1_AP[Res_1_AP != Res_1_AP] = 0
    Res_0_AP[Res_0_AP != Res_0_AP] = 0

    # TEST
    pred = Res.clamp(min=args.thr) / Res.max()
    pred = pred.view(-1).data.cpu().numpy()
    target = labels.view(-1).data.cpu().numpy()
    # print("target", target.shape)

    output = torch.cat((Res_0, Res_1), 1)

    output_AP = torch.cat((Res_0_AP, Res_1_AP), 1)

    if args.save_img:
        # Save predicted mask
        mask = F.interpolate(Res_1, [64, 64], mode='bilinear')
        mask = mask[0].squeeze().data.cpu().numpy()
        # mask = Res_1[0].squeeze().data.cpu().numpy()
        mask = 255 * mask
        mask = mask.astype('uint8')
        imageio.imsave(os.path.join(args.exp_img_path, 'mask_' + str(index) + '.jpg'), mask)

        relevance = F.interpolate(Res, [64, 64], mode='bilinear')
        relevance = relevance[0].permute(1, 2, 0).data.cpu().numpy()
        hm = np.sum(relevance, axis=-1)
        maps = (render.hm_to_rgb(hm, scaling=3, sigma=1, cmap='seismic') * 255).astype(np.uint8)
        imageio.imsave(os.path.join(args.exp_img_path, 'heatmap_' + str(index) + '.jpg'), maps)

    # Evaluate Segmentation
    batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
    batch_ap, batch_f1 = 0, 0

    # Segmentation resutls
    correct, labeled = batch_pix_accuracy(output[0].data.cpu(), labels[0])
    inter, union = batch_intersection_union(output[0].data.cpu(), labels[0], 2)

    batch_correct += correct
    batch_label += labeled
    batch_inter += inter
    batch_union += union
    ap = np.nan_to_num(get_ap_scores(output, labels))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1 = np.nan_to_num(get_f1_scores(output[0, 1].data.cpu(), labels[0]))
    batch_ap += ap
    batch_f1 += f1

    return batch_correct, batch_label, batch_inter, batch_union, batch_ap, batch_f1, pred, target


total_inter, total_union, total_correct, total_label = np.int64(0), np.int64(0), np.int64(0), np.int64(0)
total_ap, total_f1 = [], []

predictions, targets = [], []
for batch_idx, (image, labels) in enumerate(iterator):
    if args.method == "blur":
        images = (image[0].to(device), image[1].to(device))
    else:
        images = image.to(device)
    labels = labels.to(device)

    correct, labeled, inter, union, ap, f1, pred, target = eval_batch(images, labels, model, batch_idx, args.acc_cutoff, args.threshold, args.kappa)

    if type(correct) is int:
        continue

    predictions.append(pred)
    targets.append(target)

    total_correct += correct.astype('int64')
    total_label += labeled.astype('int64')
    total_inter += inter.astype('int64')
    total_union += union.astype('int64')
    total_ap += [ap]
    total_f1 += [f1]
    pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
    IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
    mIoU = IoU.mean()
    mAp = np.mean(total_ap)
    mF1 = np.mean(total_f1)
    iterator.set_description('pixAcc' + args.method + ': %.4f, mIoU: %.4f, mAP: %.4f, mF1: %.4f' % (pixAcc, mIoU, mAp, mF1))

    if batch_idx == 100:
        break

predictions = np.concatenate(predictions)
targets = np.concatenate(targets)
pr, rc, thr = precision_recall_curve(targets, predictions)
np.save(os.path.join(saver.experiment_dir, 'precision.npy'), pr)
np.save(os.path.join(saver.experiment_dir, 'recall.npy'), rc)

plt.figure()
plt.plot(rc, pr)
plt.savefig(os.path.join(saver.experiment_dir, 'PR_curve_{}.png'.format(args.method)))

txtfile = os.path.join(saver.experiment_dir, 'result_mIoU_%.4f.txt' % mIoU)
# txtfile = 'result_mIoU_%.4f.txt' % mIoU
fh = open(txtfile, 'w')
print("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
print("Mean AP over %d classes: %.4f\n" % (2, mAp))
print("Mean F1 over %d classes: %.4f\n" % (2, mF1))

fh.write("Mean IoU over %d classes: %.4f\n" % (2, mIoU))
fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
fh.write("Mean AP over %d classes: %.4f\n" % (2, mAp))
fh.write("Mean F1 over %d classes: %.4f\n" % (2, mF1))
fh.close()
