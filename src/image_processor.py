import cv2
import torch
import numpy as np
from torch.autograd import Variable

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

def extract_patch(image, window_size, stride):
    height, width = image.size

    patch_images = []
    for y in range(0, height - window_size + 1, stride):
        for x in range(0, width - window_size + 1, stride):
            patch = image.crop((x, y, x+window_size, y+window_size))
            patch_images.append(patch)

    return patch_images


def merge_patch(patch_images, window_size, effective_size, w, h):
    for i in range(h):
        for j in range(w):
            patch = patch_images[i*h + j]

            upper_margin = window_size - effective_size
            left_margin = window_size - effective_size

            if i == 0:
                upper_margin = 0
            
            if j == 0:
                left_margin = 0

            upper = upper_margin
            bottom = upper_margin + effective_size

            left = left_margin
            right = left_margin + effective_size

            effective_region = patch[upper:bottom, left:right]

            if j == 0:
                horizontal_regions = effective_region
            else:
                horizontal_regions = np.concatenate([horizontal_regions, effective_region], 1)

        if i == 0:
            image = horizontal_regions
        else:
            image = np.concatenate([image, horizontal_regions], 0)

    return image

def gammma_contrast(x, r):
    x = np.float64(x)
    y = x / 255.
    y = y **(1/r)
    return np.uint8(255*y)


def make_residual_map(x, y):
    residual_map = cv2.absdiff(x, y)
    residual_map = cv2.cvtColor(residual_map, cv2.COLOR_BGR2GRAY)
    residual_map = gammma_contrast(residual_map, 3)
    return residual_map

def make_heatmap(x):
    return cv2.applyColorMap(x, cv2.COLORMAP_JET)

def make_ssim_map(x, y):
    h, w = x.shape[:2]

    ssim_map = []

    for n,i in enumerate(range(0, h - 11 + 1)):
        for m,j in enumerate(range(0, w - 11 + 1)):
            x_patch = x[i:i+11, j:j+11, :]
            y_patch = y[i:i+11, j:j+11, :]

            x_patch = torch.from_numpy(x_patch.astype(np.float32)).clone()
            y_patch = torch.from_numpy(y_patch.astype(np.float32)).clone()

            x_patch = torch.unsqueeze(x_patch, 0)
            y_patch = torch.unsqueeze(y_patch, 0)

            x_patch = x_patch.permute(0, 3, 1, 2)
            y_patch = y_patch.permute(0, 3, 1, 2)

            ssim_map.append(1 - ssim(Variable(x_patch), Variable(y_patch), data_range=255, size_average=True))

    ssim_map = np.clip((np.array(ssim_map).reshape([n+1, m+1]) * 255), 0, 255).astype(np.uint8)

    ssim_map = cv2.resize(ssim_map, (h, w))

    return ssim_map







