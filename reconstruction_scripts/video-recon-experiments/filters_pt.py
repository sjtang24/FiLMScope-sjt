import torch
import torch.nn as nn 
import torch.nn.functional as F


def median_filter(img, size, unfold):
    """
    Perform median filtering on input image (img) with some kernel size (size).
    For integration with other kernel filters, user must provide an object of the 
    Unfold class which essentially extracts the associated patches of the image as
    desired. We have padding of size (kernel_h // 2, kernel_w // 2), replicating the
    pixel values at the boundary.

    INPUTS
    img: tensor of size (1, 1, H, W), which we expect because we process frames one at a time. 
    size: tuple of form (kernel_height, kernel_width)
    unfold: should be an instance of Unfold(size, ...default parameters...)

    OUTPUTS
    filtered_image: tensor of size (1, 1, H, W)
    """
    kernel_h, kernel_w = size
    H, W = img.shape[-2], img.shape[-1]
    pad_h, pad_w = kernel_h // 2, kernel_w // 2

    img = F.pad(img, pad=(pad_w, pad_w, pad_h, pad_h), mode='replicate')
    medians = unfold(img).median(dim = 1).values
    filtered_image = medians.view(1, 1, H, W)
    return filtered_image


