import torch

downsample = 2
x = torch.Tensor([[1, 2], [3, 4]])
print(x.repeat_interleave(downsample, dim = 0).repeat_interleave(downsample, dim = 1))