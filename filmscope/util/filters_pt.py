import torch.nn.functional as F

def median_filter(img, size, unfold):
    kernel_h, kernel_w = size
    H, W = img.shape[-2], img.shape[-1]
    pad_h, pad_w = kernel_h // 2, kernel_w //2 

    img = F.pad(img, pad = (pad_w, pad_w, pad_h, pad_h), mode = 'reflect')
    medians = unfold(img).median(dim = 1).values
    filtered_image = medians.view(1, 1, H, W)
    return filtered_image