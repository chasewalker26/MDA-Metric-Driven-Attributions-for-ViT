import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
from fpdf import FPDF
from time import time
from skimage.segmentation import slic
from skimage.util import img_as_float

os.sys.path.append(os.path.dirname(os.path.abspath('..')))

from util import model_utils
from util.test_methods import MASTestFunctions as MAS
from util.visualization import attr_to_subplot
from util.attribution_methods import MDAFunctions
from util.attribution_methods.VIT_LRP.ViT_explanation_generator import Baselines, LRP
from util.attribution_methods.TIS import TIS
from util.attribution_methods.ViT_CX.ViT_CX import ViT_CX

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_tiny_patch16_224 as vit_new_tiny_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_tiny_patch16_224 as vit_LRP_tiny_16
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch16_224 as vit_new_base_16
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch16_224 as vit_LRP_base_16
    from util.attribution_methods.VIT_LRP.ViT_new_timm import vit_base_patch32_224 as vit_new_base_32
    from util.attribution_methods.VIT_LRP.ViT_LRP_timm import vit_base_patch32_224 as vit_LRP_base_32

model = None
normalize = transforms.Normalize(
    (0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5)
)

# standard ImageNet normalization
transform_normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],
    std=[0.5, 0.5, 0.5]
)

resize = transforms.Resize((224, 224), antialias = True)

def get_score(image, attribution, scoring_method, device, max_batch_size):
    saliency_map_test = np.abs(np.sum(attribution, axis = 2))
    _, corrected_score, _, _, _ = scoring_method.single_run(image, saliency_map_test, device, max_batch_size)
    return str(round(MAS.auc(corrected_score).item(), 3))

def run_and_save_tests(img_hw, transform, explainer, LRP_explainer, batch_size, model, model_name, device, imagenet):
    if model_name == "VIT_base_8":
        segment_count = 28 ** 2
    elif model_name == "VIT_tiny_16":
        segment_count = 14 ** 2
    elif model_name == "VIT_base_16":
        segment_count = 14 ** 2
    elif model_name == "VIT_base_32":
        segment_count = 7 ** 2

    # classes excluded due to undesirable input images
    excluded_classes = [434, 435, 436, 638, 639, 842]

    # this tracks images that are classified correctly
    correctly_classified = np.loadtxt("../../util/class_maps/ImageNet/correctly_classified_" + model_name + ".txt").astype(np.int64)

    images_used = 0
    images_seen = 0
    images_per_page = 17
    num_pages = 5
    total_images = images_per_page * num_pages
    images_per_class = int(np.ceil(total_images / 1000))
    classes_used = [0] * 1000

    # make the temp image folder if it doesn't exist
    if not os.path.exists("temp_folder_attr"):
        os.makedirs("temp_folder_attr")

    pdf = FPDF(format = "letter", unit="in")

    images = sorted(os.listdir(imagenet))

    norm = "absolute"
    alpha = 0.6
    cmap = 'jet'

    plt.rcParams.update({'figure.dpi': 50})

    for j in range(num_pages):
        plt.rcParams.update({'font.size': 35})
        fig, axs = plt.subplots(17, 11, figsize = (55, 85 + 2))

        i = 0
        while (i != images_per_page):
            image = images[images_seen]
            images_seen += 1

            # check if the current image is an invalid image for testing, 0 indexed
            image_num = int((image.split("_")[2]).split(".")[0]) - 1
            # check if the current image is an invalid image for testing
            if correctly_classified[image_num] == 0:
                continue

            image_path = imagenet + "/" + image
            PIL_img = Image.open(image_path)
            trans_img = transform(PIL_img)
            # put the image in form needed for prediction
            input_tensor = transform_normalize(trans_img)
            input_tensor = torch.unsqueeze(input_tensor, 0).to(device)

            # only rgb images can be classified
            if trans_img.shape != (3, img_hw, img_hw):
                continue

            target_class = model_utils.getClass(input_tensor, model, device)
            if target_class in excluded_classes:
                continue
        
            # open the class list so the detected class string can be returned for printing
            with open('../../util/class_maps/ImageNet/imagenet_classes.txt') as f:
                classes = [line.strip() for line in f.readlines()]
            class_name = classes[target_class]

            if model_utils.getPrediction(input_tensor, model, device, target_class)[0] * 100 < 60:
                continue

            # Track which classes have been used
            if classes_used[target_class] == images_per_class:
                continue
            else:
                classes_used[target_class] += 1       

            klen = 31
            ksig = 31
            kern = MAS.gkern(klen, ksig)
            blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
            blur_pred = model_utils.getPrediction(blur(input_tensor.cpu()).to(device), model, device, target_class)[0] * 100
            # choose a blurring kernel that results in a softmax score of 1% or less 
            while blur_pred > 1 and klen <= 101:
                klen += 2
                ksig += 2
                kern = MAS.gkern(klen, ksig)
                blur = lambda x: nn.functional.conv2d(x, kern, padding = klen // 2)
                blur_pred = model_utils.getPrediction(blur(input_tensor.cpu()).to(device), model, device, target_class)[0] * 100

            if klen > 101:
                classes_used[target_class] -= 1
                continue

            # MAS_insertion = MAS.MASMetric(model, img_hw * img_hw, 'ins', img_hw, substrate_fn = blur)
            # MAS_deletion = MAS.MASMetric(model, img_hw * img_hw, 'del', img_hw, substrate_fn = torch.zeros_like)

            start = time()

            print(model_name + ", image: " + image + " " + str(images_used + 1) + "/" + str(total_images))

            ########  IG  ########
            _, IG, _, _, _ = explainer.generate_transition_attention_maps(input_tensor, target_class, start_layer = 0, device = device)
            IG = resize(IG.cpu().detach())
            ig = IG.permute(1, 2, 0).numpy() * np.ones((224, 224, 3))

            ########  GC  ########
            attr = explainer.generate_cam_attn(input_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            gc = attr.permute(1, 2, 0).numpy() * np.ones((224, 224, 3))

            ########  LRP  ########
            attr = LRP_explainer.generate_LRP(input_tensor, target_class, method="transformer_attribution", start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            lrp = (attr).permute(1, 2, 0).numpy() * np.ones((224, 224, 3))

            ########  T Attn  ########
            _, _, attr, _, _ = explainer.generate_transition_attention_maps(input_tensor, target_class, start_layer = 0, device = device)
            attr = resize(attr.cpu().detach())
            trans_attn = (attr).permute(1, 2, 0).numpy() * np.ones((224, 224, 3))

            ########  Bi Attn  ########
            attr, _ = explainer.bidirectional(input_tensor, target_class, device = device)
            attr = resize(attr.cpu().detach())
            bi_attn = (attr).permute(1, 2, 0).numpy() * np.ones((224, 224, 3))

            ########  TIS  ########
            saliency_method = TIS(model, batch_size=64)
            saliency_map = saliency_method(input_tensor, class_idx=target_class).cpu()
            tis = resize(saliency_map.reshape((-1, saliency_map.shape[0], saliency_map.shape[1])).detach()).permute(1, 2, 0).cpu().numpy() * np.ones((224, 224, 3))

            ########  ViT-CX  ########
            target_layer = model.blocks[-1].norm1
            result, features = ViT_CX(model, input_tensor, target_layer, gpu_batch=1, device = device)
            saliency_map = (result.reshape((224, 224, 1)) * torch.ones((224, 224, 3))).cpu().detach().numpy()
            vit_cx = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

            ########  MDA  ########
            segment_img = np.transpose(trans_img.squeeze().detach().numpy(), (1, 2, 0))
            segment_img = img_as_float(segment_img)
            segments = torch.tensor(slic(segment_img, n_segments=segment_count, compactness=10000, start_label=0), dtype = int)
            upsize = transforms.Resize((224, 224), transforms.InterpolationMode.NEAREST_EXACT)
            downsize = transforms.Resize(int(segment_count ** (1/2)))
            saliency_map_segmented = upsize(downsize(torch.tensor(bi_attn, dtype = torch.float).permute((2, 0, 1)))).permute((1, 2, 0))
            _, _, order_a, MR_ins = MDAFunctions.find_insertion_patches(input_tensor.cpu(), saliency_map_segmented, segments, blur, segment_count, type = 1, model = model, device = device, img_hw = img_hw, max_batch_size = 5)
            end_index = np.where(MR_ins >= 0.9)[0][0]
            MDA, MDA_smooth, MDA_b, MDA_smooth_b,  MDA_c, MDA_smooth_c,  _ = MDAFunctions.find_deletion_patches(input_tensor.cpu(), segments, saliency_map_segmented, order_a[0 : end_index + 1], blur, segment_count, model, device, img_hw, max_batch_size = batch_size)

            input_tensor = input_tensor.cpu()

            axs[i, 0].set_ylabel(class_name.replace(" ", "\n"))

            if i == 0:
                attr_to_subplot(trans_img, "Input", axs[i, 0], original_image=True)
                attr_to_subplot(ig, "IG", axs[i, 1], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(gc, "GC", axs[i, 2], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(trans_attn, "T-Attn", axs[i, 3], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(lrp, "T-Attr", axs[i, 4], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(bi_attn, "Bi-Attn", axs[i, 5], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(tis, "TIS", axs[i, 6], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(vit_cx, "ViT-CX", axs[i, 7], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth, "MDA $\\gamma = 0.0$", axs[i, 8], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth_b, "MDA $\\gamma = 0.5$", axs[i, 9], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth_c, "MDA $\\gamma = 1.0$", axs[i, 10], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
            else:
                attr_to_subplot(trans_img, "", axs[i, 0], original_image=True)
                attr_to_subplot(ig, "", axs[i, 1], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(gc, "", axs[i, 2], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(trans_attn, "", axs[i, 3], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(lrp, "", axs[i, 4], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(bi_attn, "", axs[i, 5], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(tis, "", axs[i, 6], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(vit_cx, "", axs[i, 7], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth, "", axs[i, 8], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth_b, "", axs[i, 9], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)
                attr_to_subplot(MDA_smooth_c, "", axs[i, 10], cmap = cmap, norm = norm, blended_image = trans_img, alpha = alpha)

            print(time() - start)

            images_used += 1
            i += 1
    
        print("Saving Page " + str(j + 1) + "/" + str(num_pages))
        # save the figure for the current attribution function
        plt.figure(fig)
        plt.subplots_adjust(wspace = 0.05)
        plt.savefig("temp_folder_attr/" + str(j) + ".png", bbox_inches='tight', transparent = "True")
        fig.clear()
        plt.close(fig)

    print("Assembling PDF")

    # (LM x TM x W x H)
    for j in range(num_pages):
        pdf.add_page()
        pdf.image("temp_folder_attr/" + str(j) + ".png", 1.5, 1, 5.5, 8.7)

    # make the test folder if it doesn't exist
    folder = "test_results/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    pdf.output(folder + model_name + "_" + "MDA_supplementary" + ".pdf", "F")
    pdf.close()

    print("Cleaning Up")
    # clear the folder that held the images
    dir = 'temp_folder_attr'
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    os.rmdir(dir)

    return

def main(FLAGS):
    device = 'cuda:' + str(FLAGS.cuda_num) if torch.cuda.is_available() else 'cpu'

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

    run_and_save_tests(img_hw, transform, explainer, LRP_explainer, batch_size, model, model_name, device, FLAGS.imagenet)

if __name__ == "__main__":
    # Set parameters for Sparse Autoencoder
    parser = argparse.ArgumentParser('Make a pdf of comparing calibration methods')
    parser.add_argument('--cuda_num',
                        type=int, default = 0,
                        help='The number of the GPU you want to use.')
    parser.add_argument('--model',
                    type = str, default = "VIT_base_16",
                    help = 'Name of the model to use: VIT_tiny_16, VIT_base_32, VIT_base_16.')
    parser.add_argument('--imagenet',
                type = str, default = "imagenet",
                help = 'The path to your 2012 imagenet vlaidation set. Images in this folder should have the name structure: "ILSVRC2012_val_00000001.JPEG".')
    
    FLAGS, unparsed = parser.parse_known_args()
    
    main(FLAGS)