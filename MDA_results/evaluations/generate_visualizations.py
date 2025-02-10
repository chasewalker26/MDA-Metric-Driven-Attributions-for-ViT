import os
from tqdm import tqdm
import h5py

import argparse
import warnings
from torch import nn
import torch

# Import saliency methods and models
from utils.misc_functions import *

from torchvision.datasets import ImageNet
from skimage.segmentation import slic
from skimage.util import img_as_float

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.attribution_methods import MDAFunctions
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods.TIS import TIS
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX
from util.test_methods import MASTestFunctions as MAS

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_base_patch16_224_LRP

    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_base_patch32_224_LRP

resize = transforms.Resize((224, 224), antialias = True)


inv_normalize = transforms.Normalize(
    mean=[-1, -1, -1],
    std=[2, 2, 2]
)

def normalize(tensor,
              mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return tensor


def compute_saliency_and_save(args):
    first = True
    with h5py.File(os.path.join(args.method_dir, 'results.hdf5'), 'a') as f:
        data_cam = f.create_dataset('vis',
                                    (1, 1, 224, 224),
                                    maxshape=(None, 1, 224, 224),
                                    dtype=np.float32,
                                    compression="gzip")
        data_image = f.create_dataset('image',
                                      (1, 3, 224, 224),
                                      maxshape=(None, 3, 224, 224),
                                      dtype=np.float32,
                                      compression="gzip")
        data_target = f.create_dataset('target',
                                       (1,),
                                       maxshape=(None,),
                                       dtype=np.int32,
                                       compression="gzip")
        
        iter_object = enumerate(sample_loader)
        for i in tqdm(range(1000)):
            batch_idx, (data, target) = next(iter_object)

        # for batch_idx, (data, target) in enumerate(tqdm(sample_loader)):
            if first:
                first = False
                data_cam.resize(data_cam.shape[0] + data.shape[0] - 1, axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0] - 1, axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0] - 1, axis=0)
            else:
                data_cam.resize(data_cam.shape[0] + data.shape[0], axis=0)
                data_image.resize(data_image.shape[0] + data.shape[0], axis=0)
                data_target.resize(data_target.shape[0] + data.shape[0], axis=0)

            # Add data
            data_image[-data.shape[0]:] = data.data.cpu().numpy()
            data_target[-data.shape[0]:] = target.data.cpu().numpy()

            target = target.to(device)

            data = normalize(data)
            data = data.to(device)
            data.requires_grad_()

            batch_size = data.shape[0]

            index = None
            if args.vis_class == 'target':
                index = target

            target_class = model_utils.getClass(data, model, device)
            original_pred = model_utils.getPrediction(data, model, device, target_class)[0] * 100
            if original_pred < 60:
                continue

            klen = 1
            ksig = 1
            kern = MAS.gkern(klen, ksig)
            blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
            blur_pred = model_utils.getPrediction(blur(data.cpu()).to(device), model, device, target_class)[0] * 100
            # choose a blurring kernel that results in a softmax score of 1% or less 
            while blur_pred > 1 and klen <= 101:
                klen += 2
                ksig += 2
                kern = MAS.gkern(klen, ksig)
                blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
                blur_pred = model_utils.getPrediction(blur(data.cpu()).to(device), model, device, target_class)[0] * 100

            if klen > 101:
                continue

            if (args.method == 'IG'):    
                _, IG, _, _, _ = baselines.generate_transition_attention_maps(data, target_class, start_layer = 0, device = device)
                IG = resize(IG.cpu().detach())
                Res = (IG).permute(1, 2, 0)
                Res = Res.reshape(batch_size, 1, 224, 224)

            elif (args.method == 'GC'):    
                attr = baselines.generate_cam_attn(data, target_class, device = device)
                attr = resize(attr.cpu().detach())
                Res = (attr).permute(1, 2, 0)
                Res = Res.reshape(batch_size, 1, 224, 224)

            elif (args.method == 'LRP'):
                attr = lrp.generate_LRP(data, target_class, method="transformer_attribution", start_layer = 0, device = device)
                attr = resize(attr.cpu().detach())
                Res = (attr).permute(1, 2, 0)
                Res = Res.reshape(batch_size, 1, 224, 224)

            elif (args.method == 'Transition_attn'):
                _, _, attr, _, _ = baselines.generate_transition_attention_maps(data, target_class, start_layer = 0, device = device)
                attr = resize(attr.cpu().detach())
                Res = (attr).permute(1, 2, 0)
                Res = Res.reshape(batch_size, 1, 224, 224)

            elif (args.method == 'Bidirectional'):
                attr, _ = baselines.bidirectional(data, target_class, device = device)
                attr = resize(attr.cpu().detach())
                Res = (attr).permute(1, 2, 0)
                Res = Res.reshape(batch_size, 1, 224, 224)

            elif (args.method == "TIS"):
                saliency_method = TIS(model, batch_size=64)
                saliency_map = saliency_method(data, class_idx=target_class).cpu()
                Res = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 1))
                Res = Res.reshape(batch_size, 1, 224, 224)
                Res = torch.from_numpy(Res)

            elif (args.method == "VIT_CX"):
                target_layer = model.blocks[-1].norm1
                result, _ = ViT_CX(model, data, target_layer, gpu_batch=1, device = device)
                saliency_map = (result.reshape((224, 224, 1)) * torch.ones((224, 224, 1))).cpu().detach().numpy()
                Res = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
                Res = Res.reshape(batch_size, 1, 224, 224)
                Res = torch.from_numpy(Res)

            elif (args.method == 'Calibrate_Ins'):
                trans_img = inv_normalize(data.squeeze())
                segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
                segment_img = img_as_float(segment_img)
                segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
                attr_attr, _ = baselines.bidirectional(data, target_class, device = device)
                reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
                upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias=True)
                downsize = transforms.Resize(int(segment_count ** (1/2)), antialias=True)
                saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
                _, saliency_map_smooth, _, _ = MDAFunctions.find_insertion_patches(data.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25, cutoff = 1)
                saliency_map_smooth = torch.tensor(saliency_map_smooth)
                Res = saliency_map_smooth.mean(dim=-1).reshape(batch_size, 1, 224, 224)

            elif (args.method == 'Calibrate_Del'):
                trans_img = inv_normalize(data.squeeze())
                segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
                segment_img = img_as_float(segment_img)
                segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
                attr_attr, _ = baselines.bidirectional(data, target_class, device = device)
                reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
                upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias=True)
                downsize = transforms.Resize(int(segment_count ** (1/2)), antialias=True)
                saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
                _, saliency_map = MDAFunctions.find_deletion_patches(data.cpu(), segments, saliency_map_segmented, torch.tensor([]), blur, segment_count, model, device, 224, max_batch_size = 25)
                upsize = transforms.Resize(224, antialias=True)
                small_side = int(np.ceil(np.sqrt(segment_count)))
                downsize = transforms.Resize(small_side, antialias=True)
                Res = upsize(downsize(saliency_map.permute((2, 0, 1)))).permute((1, 2, 0))
                Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)

            elif (args.method == 'Calibrate_Sparse'):
                trans_img = inv_normalize(data.squeeze())
                segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
                segment_img = img_as_float(segment_img)
                segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
                attr_attr, _ = baselines.bidirectional(data, target_class, device = device)
                reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
                upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias=True)
                downsize = transforms.Resize(int(segment_count ** (1/2)), antialias=True)
                saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
                _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(data.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
                end_index = np.where(MR_ins >= 0.9)[0][0]
                Res, _ = MDAFunctions.find_deletion_patches(data.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25)
                Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)


            elif (args.method == 'Calibrate_Sparse_Smooth'):
                trans_img = inv_normalize(data.squeeze())
                segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
                segment_img = img_as_float(segment_img)
                segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
                attr_attr, _ = baselines.bidirectional(data, target_class, device = device)
                reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
                upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias = True)
                downsize = transforms.Resize(int(segment_count ** (1/2)), antialias = True)
                saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
                _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(data.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
                end_index = np.where(MR_ins >= 0.9)[0][0]
                Res, _ = MDAFunctions.find_deletion_patches(data.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25)
                upsize = transforms.Resize(224, antialias=True)
                small_side = int(np.ceil(np.sqrt(segment_count)))
                downsize = transforms.Resize(small_side, antialias=True)
                Res = upsize(downsize(Res.permute((2, 0, 1)))).permute((1, 2, 0))
                Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)


            elif (args.method == 'Calibrate_Ordered'):
                trans_img = inv_normalize(data.squeeze())
                segment_img = np.transpose(trans_img.squeeze().detach().cpu().numpy(), (1, 2, 0))
                segment_img = img_as_float(segment_img)
                segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
                attr_attr, _ = baselines.bidirectional(data, target_class, device = device)
                reference_attr = resize(attr_attr.cpu().detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))
                upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT, antialias=True)
                downsize = transforms.Resize(int(segment_count ** (1/2)), antialias=True)
                saliency_map_segmented = upsize(downsize(torch.tensor(reference_attr, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
                _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(data.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = 224, max_batch_size = 25)
                end_index = np.where(MR_ins >= 0.9)[0][0]
                _, Res = MDAFunctions.find_deletion_patches(data.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, 224, max_batch_size = 25, kappa=-1)
                Res = Res.mean(dim=-1).reshape(batch_size, 1, 224, 224)

            Res = (Res - Res.min()) / (Res.max() - Res.min())

            data_cam[-data.shape[0]:] = Res.data.cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation')
    parser.add_argument('--batch-size', type=int,
                        default=1,
                        help='')
    parser.add_argument('--method', type=str,
                        default='grad_rollout',
                        help='')
    parser.add_argument('--model', type=str,
                    default='VIT_base_16',
                    help='VIT_base_16 or VIT_base_32')
    parser.add_argument('--lmd', type=float,
                        default=10,
                        help='')
    parser.add_argument('--vis-class', type=str,
                        default='top',
                        choices=['top', 'target', 'index'],
                        help='')
    parser.add_argument('--class-id', type=int,
                        default=0,
                        help='')
    parser.add_argument('--cls-agn', action='store_true',
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
    parser.add_argument('--imagenet-validation-path', type=str,
                        required=True,
                        help='')
    args = parser.parse_args()

    # PATH variables
    PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
    os.makedirs(os.path.join(PATH, 'visualizations'), exist_ok=True)

    try:
        os.remove(os.path.join(PATH, 'visualizations/{}/{}/{}/results.hdf5'.format(args.model,args.method,
                                                                                args.vis_class)))
    except OSError:
        pass


    os.makedirs(os.path.join(PATH, 'visualizations/{}/{}'.format(args.model,args.method)), exist_ok=True)
    if args.vis_class == 'index':
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}_{}'.format(args.model,args.method,
                                                                        args.vis_class,
                                                                        args.class_id)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}_{}'.format(args.model,args.method,
                                                                              args.vis_class,
                                                                              args.class_id))
    else:
        ablation_fold = 'ablation' if args.is_ablation else 'not_ablation'
        os.makedirs(os.path.join(PATH, 'visualizations/{}/{}/{}/{}'.format(args.model,args.method,
                                                                     args.vis_class, ablation_fold)), exist_ok=True)
        args.method_dir = os.path.join(PATH, 'visualizations/{}/{}/{}/{}'.format(args.model,args.method,
                                                                           args.vis_class, ablation_fold))

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

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

    # Dataset loader for sample images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    imagenet_ds = ImageNet(args.imagenet_validation_path, split='val', transform=transform)
    sample_loader = torch.utils.data.DataLoader(
        imagenet_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    compute_saliency_and_save(args)
