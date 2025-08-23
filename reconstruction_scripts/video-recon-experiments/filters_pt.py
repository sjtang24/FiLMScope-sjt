import torch
import torch.nn as nn 
import torch.nn.functional as F


def median_filter(img, size, unfold):
    # precondition (size == unfold.kernel_size)
    kernel_h, kernel_w = size
    H, W = img.shape[-2], img.shape[-1]
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    img = F.pad(img, pad=(pad_w, pad_w, pad_h, pad_h), mode='replicate')
    medians = unfold(img).median(dim = 1).values
    filtered = medians.view(1, 1, H, W)
    return filtered
    #print(patches.shape)"""



