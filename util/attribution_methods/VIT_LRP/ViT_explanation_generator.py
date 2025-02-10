# https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py


import argparse
import torch
import numpy as np
from numpy import *
import torch.nn as nn

# compute rollout between attention layers with residual connection modeling
# https://github.com/samiraabnar/attention_flow
def compute_rollout_attention(all_layer_matrices, start_layer = 0):
    num_tokens = all_layer_matrices[0].shape[1]
    batch_size = all_layer_matrices[0].shape[0]
    eye = torch.eye(num_tokens).expand(batch_size, num_tokens, num_tokens).to(all_layer_matrices[0].device)

    # add residual connection modeling to all matrices
    all_layer_matrices = torch.stack(all_layer_matrices)
    num_blocks = all_layer_matrices.shape[0]

    matrices_aug = all_layer_matrices + eye.unsqueeze(0)

    # normalize all the matrices, making residual connection addition equal to 0.5*A + 0.5*I
    matrices_aug = matrices_aug / matrices_aug.sum(dim=-1, keepdim=True)

    # perform rollout
    joint_attention = matrices_aug[start_layer]
    for i in range(start_layer + 1, num_blocks):
        joint_attention = matrices_aug[i].bmm(joint_attention)

    return joint_attention, matrices_aug

# https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py
class LRP:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate_LRP(self, input, target_class, method="transformer_attribution", is_ablation=False, start_layer=0, withgrad = True, device = "cuda:0"):
        output = self.model(input)
        kwargs = {"alpha": 1}
        if target_class == None:
            target_class = np.argmax(output.cpu().data.numpy(), axis=-1)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0, target_class] = 1
        one_hot_vector = one_hot
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        num_blocks = len(self.model.blocks)

        attr = self.model.relprop(torch.tensor(one_hot_vector).to(input.device), method=method, is_ablation=is_ablation,
                                  start_layer=start_layer, end_layer=num_blocks, withgrad=withgrad, **kwargs)
        
        patches_per_side = int(np.sqrt(attr.shape[-1]))

        return attr.reshape(-1, patches_per_side, patches_per_side)

class Baselines:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    # https://github.com/hila-chefer/Transformer-Explainability/blob/main/baselines/ViT/ViT_explanation_generator.py
    def generate_cam_attn(self, input, target_class, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()
        input.requires_grad = False

        #################### attn
        grad = self.model.blocks[-1].attn.get_attn_gradients()
        patches_per_side = int(np.sqrt(grad.shape[-1] - 1))
        cam = self.model.blocks[-1].attn.get_attention_map()[0, :, 0, 1:].reshape(-1, patches_per_side, patches_per_side)

        grad = grad[0, :, 0, 1:].reshape(-1, patches_per_side, patches_per_side)
        cam = (cam * grad).mean(0).clamp(min=0)
        cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam.unsqueeze(0)

    # https://github.com/XianrenYty/Transition_Attention_Maps
    def generate_transition_attention_maps(self, input, target_class, start_layer = 0, steps = 20, with_integral = True, first_state = False, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()

        b, h, s, _ = self.model.blocks[-1].attn.get_attention_map().shape

        num_blocks = len(self.model.blocks)

        # states is CLS token row
        states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        for i in range(start_layer, num_blocks)[::-1]:
            attn = self.model.blocks[i].attn.get_attention_map().mean(1)

            states_ = states
            # states column vector MVM w self-attn 
            states = torch.einsum('biw, bwh->h', states, attn).reshape(b, 1, s)

            # add residual
            states += states_

        total_gradients = torch.zeros(b, h, s, s).to(device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backprop
            output = self.model(data_scaled, register_hook=True)
            score = output[0][target_class].sum()
            score.backward()

            # call grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients

        if with_integral:
            W_state = (total_gradients / steps).clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        else:
            W_state = self.model.blocks[-1].attn.get_attn_gradients().clamp(min=0).mean(1)[:, 0, :].reshape(b, 1, s)
        
        if first_state:
            states = self.model.blocks[-1].attn.get_attention_map().mean(1)[:, 0, :].reshape(b, 1, s)
        
        final = states * W_state
        input.requires_grad = False
        
        patches_per_side = int(np.sqrt(s - 1))

        return states[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), W_state[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), final[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), self.model.blocks[-1].attn.get_attention_map().mean(1)[0, 0, 1:], gradients

    # https://github.com/jiaminchen-1031/transformerinterp/blob/master/ViT/baselines/ViT/ViT_explanation_generator.py
    def bidirectional(self, input, target_class, steps=20, start_layer=4, samples=20, noise=0.2, mae=False, dino=False, ssl=False, device = "cuda:0"):
        input.requires_grad = True
        output = self.model(input.to(device), register_hook=True)
        score = output[0][target_class].sum()
        score.backward()
        
        b, num_head, num_tokens, _ = self.model.blocks[-1].attn.get_attention_map().shape

        R = torch.eye(num_tokens, num_tokens).expand(b, num_tokens, num_tokens).to(device)
        for nb, blk in enumerate(self.model.blocks):
            if nb < start_layer - 1:
                continue
            
            grad = blk.attn.get_attn_gradients()
            grad = grad.reshape(-1, grad.shape[-2], grad.shape[-1])
            cam = blk.attn.get_attention_map()
            cam = cam.reshape(-1, cam.shape[-2], cam.shape[-1])

            Ih = torch.mean(torch.matmul(cam.transpose(-1,-2), grad).abs(), dim=(-1,-2))
            Ih = Ih/torch.sum(Ih)

            cam = torch.matmul(Ih, cam.reshape(num_head,-1)).reshape(num_tokens,num_tokens)
            
            R = R + torch.matmul(cam.to(device), R.to(device))

        if ssl:
            if mae:
                return R[:, 1:, 1:].abs().mean(axis=1)
            elif dino:
                return (R[:, 1:, 1:].abs().mean(axis=1)+R[:, 0, 1:].abs())
            else:
                return R[:, 0, 1:].abs()
        
        # integrated gradients
        total_gradients = torch.zeros(b, num_head, num_tokens, num_tokens).to(device)
        for alpha in np.linspace(0, 1, steps):        
            # forward propagation
            data_scaled = input * alpha

            # backprop
            output = self.model(data_scaled, register_hook=True)
            score = output[0][target_class].sum()
            score.backward()

            # calc grad
            gradients = self.model.blocks[-1].attn.get_attn_gradients()
            total_gradients += gradients        
       
        W_state = (total_gradients / steps).clamp(min=0).mean(1).reshape(b, num_tokens, num_tokens)
        attr = W_state * R

        input.requires_grad = False

        patches_per_side = int(np.sqrt(num_tokens - 1))

        if mae:
            return attr[:, 1:, 1:].mean(axis=1)
        elif dino:
            return (attr[:, 1:, 1:].mean(axis=1) + attr[:, 0, 1:])
        else:
            return attr[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side), R[:, 0, 1:].reshape(-1, patches_per_side, patches_per_side)