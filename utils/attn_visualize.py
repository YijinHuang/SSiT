# ==================================================================================
# Based on https://github.com/facebookresearch/dino/blob/main/visualize_attention.py
# ==================================================================================
import os
import sys
import cv2
import random
import argparse
import colorsys

import torch
import skimage.io
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from matplotlib.patches import Polygon
from skimage.measure import find_contours
from torchvision import transforms as pth_transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vits import archs, resize_pos_embed
from funcs import load_checkpoint


parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, default='ViT-S-p16', help='Architecture (support only ViT).')
parser.add_argument('--patch-size', default=16, type=int, help='Patch resolution of the model.')
parser.add_argument('--checkpoint', default='', type=str,
    help="Path to pretrained weights to load.")
parser.add_argument("--checkpoint-key", default="base_encoder", type=str,
    help='Key to use in the checkpoint (example: "teacher")')
parser.add_argument("--image-folder", default=None, type=str, help="Path of the image to load.")
parser.add_argument("--image-size", default=1024, type=int, nargs="+", help="Resize image.")
parser.add_argument('--output-dir', default='.', help='Path where to save visualizations.')
parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
    obtained by thresholding the self-attention maps to keep xx% of the mass.""")


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def main():
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # build model
    
    model = archs[args.arch](
        pretrained=False,
        num_classes=1,
        img_size=args.image_size,
    )

    linear_key = 'head'
    checkpoint_key = args.checkpoint_key
    checkpoint = torch.load(args.checkpoint)
    load_checkpoint(model, checkpoint, checkpoint_key, linear_key)

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    mean = [0.425753653049469, 0.29737451672554016, 0.21293757855892181]
    std = [0.27670302987098694, 0.20240527391433716, 0.1686241775751114]
    transform = pth_transforms.Compose([
        pth_transforms.Resize((args.image_size, args.image_size)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(mean, std),
    ])
    for root, _, imgs in os.walk(args.image_folder):
        for img in imgs:
            name = img.split('.')[0]
            out_dir = os.path.join(args.output_dir, name)
            img_path = os.path.join(root, img)

            with open(img_path, 'rb') as f:
                raw_img = Image.open(f)
                raw_img = raw_img.convert('RGB')

            img = transform(raw_img)
            H, W = img.shape[1:]

            # make the image divisible by the patch size
            w, h = img.shape[1] - img.shape[1] % args.patch_size, img.shape[2] - img.shape[2] % args.patch_size
            img = img[:, :w, :h].unsqueeze(0)

            w_featmap = img.shape[-2] // args.patch_size
            h_featmap = img.shape[-1] // args.patch_size

            _, f = model(img.to(device))
            f = f[:, 1:]
            B, L, C = f.shape
            H = W = int(L ** 0.5)
            f = f.permute(0, 2, 1).reshape(B, C, H, W)

            attentions = model.get_last_selfattention(img.to(device))
            nh = attentions.shape[1] # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            attentions = torch.cat([attentions, attentions.mean(dim=0, keepdim=True)], dim=0)
            nh += 1

            if args.threshold is not None:
                # we keep only a certain percentage of the mass
                val, idx = torch.sort(attentions)
                val /= torch.sum(val, dim=1, keepdim=True)
                cumval = torch.cumsum(val, dim=1)
                th_attn = cumval > (1 - args.threshold)
                idx2 = torch.argsort(idx)
                for head in range(nh):
                    th_attn[head] = th_attn[head][idx2[head]]
                th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
                # interpolate
                th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=args.patch_size, mode="nearest")[0].cpu().numpy()

            # save attentions heatmaps
            os.makedirs(out_dir, exist_ok=True)
            torchvision.utils.save_image(torchvision.utils.make_grid(img, normalize=True, scale_each=True), os.path.join(out_dir, "img.png"))
            for j in range(nh):
                fname = os.path.join(out_dir, "attn-head" + str(j) + ".png")
                plt.imsave(fname=fname, arr=attentions[j], format='png')
                print(f"{fname} saved.")
            
            normalized_attn = []
            for j in range(nh-1):
                normalized_attn.append((attentions[j] - attentions[j].min()) / (attentions[j].max() - attentions[j].min()))
            normalized_attn = np.stack(normalized_attn, axis=0)

            mean_attn = np.mean(normalized_attn, axis=0)
            fname = os.path.join(out_dir, "attn-head" + "-mean.png")
            plt.imsave(fname=fname, arr=mean_attn, format='png')
            print(f"{fname} saved.")

            max_attn = np.max(normalized_attn, axis=0)
            fname = os.path.join(out_dir, "attn-head" + "-max.png")
            plt.imsave(fname=fname, arr=max_attn, format='png')
            print(f"{fname} saved.")

            if args.threshold is not None:
                image = skimage.io.imread(os.path.join(out_dir, "img.png"))
                for j in range(nh):
                    display_instances(image, th_attn[j], fname=os.path.join(out_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


if __name__ == '__main__':
    main()
