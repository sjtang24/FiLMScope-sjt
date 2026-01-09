import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .misc import tocuda
from .base_warp_functions import generate_base_grid

# heights is a tensor
# image is a tensor with shape [N, channels, x, y]
# both maps are tensors with shape [1, x, y, 2]
def generate_warp_volume(
    image, heights, warped_shift_slopes, inv_inter_camera_map, base_grid=None,
    return_grid=False
):

    B, C, H, W = image.shape
    D = heights.shape[0]

    if base_grid is None:
        base_grid = generate_base_grid((H, W))  # (1, H, W, 2)

    # Expand base grid across batch
    base_grid = base_grid.expand(B, -1, -1, -1)

    # Add inter-camera shifts
    base_grid = base_grid + inv_inter_camera_map  # (B, H, W, 2)

    # Expand along depth dimension
    base_grid = base_grid[:, None].expand(B, D, H, W, 2)

    heights = heights.view(1, D, 1, 1, 1)

    slope_shifts = warped_shift_slopes[:, None] * heights

    grid = base_grid - slope_shifts  # (B, D, H, W, 2)

    # Prepare image stack
    image_stack = image[:, None].expand(B, D, C, H, W)
    image_stack = image_stack.reshape(B * D, C, H, W)

    grid = grid.reshape(B * D, H, W, 2)

    warped_stack = F.grid_sample(
        image_stack,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False
    )

    warped_stack = warped_stack.view(B, D, C, H, W)

    if return_grid:
        return warped_stack, grid.view(B, D, H, W, 2)

    return warped_stack

# depth values should be a torch tensor
# TODO: make it possible to save to file instead of returning?
def get_ss_volume_from_dataset(dataset, batch_size=16, depth_values=None, get_squared=None):
    """print(len(dataset))
    ImageLoader = DataLoader(
        dataset,
        batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )"""

    volume = None
    if get_squared:
        volume_sq = None

    sample_cuda = tocuda(dataset.get_full_sample())
    images = sample_cuda["imgs"]
    warped_ss_maps = sample_cuda["warped_shift_slope_maps"] - torch.asarray(dataset.ref_camera_shift_slopes).cuda()
    iic_maps = sample_cuda["inv_inter_camera_maps"]

    """images = torch.unbind(images, 0)
    iic_maps = torch.unbind(iic_maps, 0)
    warped_ss_maps = torch.unbind(warped_ss_maps, 0)"""
    base_grid = dataset.base_grid

    warped_volumes = generate_warp_volume(                                       # 0.0025s/image = 0.0545
        images, depth_values, warped_ss_maps, iic_maps, base_grid
    )

    volume = warped_volumes.sum(dim = 0, keepdim = True)
    volume_sq = (warped_volumes ** 2).sum(dim = 0, keepdim = True)

    PERM = (0, 2, 1, 3, 4)
    volume = volume.permute(PERM)
    volume_sq = volume_sq.permute(PERM)

    if get_squared:
        return volume, volume_sq
    return volume

# the goal of this is to warp the "other_image" into the reference camera frame
# using the reference camera's depth map
# view_image: [batch_size, channels, height, width]
# reference_depth: [batch_size, height, width, 1]
# ref_shift_slope: [batch_size, height, width, 2]
# view_inv_inter_camera: [batch_size, height, width, 2]
# view_warped_shift_slope: [batch_size, height, width, 2]
def inverse_warping(
    view_image,
    ref_depth_est,
    ref_shift_slope,
    view_inv_inter_camera,
    view_warped_shift_slope,
    base_grid=None,
):
    ref_depth_est = ref_depth_est.unsqueeze(-1)
    # if the mask is not None, mulitiply it with the ref_depth_est

    if base_grid is None:
        base_grid = generate_base_grid(view_image.shape[2:]).cuda()

    slope_shifts = (ref_shift_slope + view_warped_shift_slope * -1) * ref_depth_est
    full_grid = base_grid + slope_shifts + view_inv_inter_camera

    # and warp
    warped_image = F.grid_sample(
        view_image, full_grid, mode="bilinear", padding_mode="zeros",
        align_corners=False
    )

    # the mask is just where "full_grid" is >1 or <-1
    grid_x = full_grid[:, :, :, [0]]
    grid_y = full_grid[:, :, :, [1]]
    mask = (grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1)
    # permute to match images shape 
    mask = mask.permute(0, 3, 1, 2)

    return warped_image, mask

# depth_map is a 2D tensor that's a height map in reference view
# height_offsets is 1D 
# everything else is same as get_ss_volume_from_dataset
# depth_map is [1, H, W] 
def get_height_aware_vol_from_dataset(
        dataset, batch_size, height_offsets, depth_map, get_squared
):
    ImageLoader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
    )
    depth_map = depth_map.unsqueeze(-1)
    volume = None
    if get_squared:
        volume_sq = None
    for sample in ImageLoader:
        sample_cuda = tocuda(sample)
        images = sample_cuda["imgs"]
        warped_ss_maps = sample_cuda["warped_shift_slope_maps"]
        iic_maps = sample_cuda["inv_inter_camera_maps"]

        images = torch.unbind(images, 0)
        iic_maps = torch.unbind(iic_maps, 0) 
        warped_ss_maps = torch.unbind(warped_ss_maps, 0)

        for image, iic_map, warped_ss_map in zip(images, iic_maps, warped_ss_maps):
            vol = get_map_offset_volume(
                image.unsqueeze(0), depth_map, height_offsets, warped_ss_map, iic_map
            )
            vol = vol.permute(1, 0, 2, 3)[None]
        
            if volume is None:
                volume = vol 
            else:
                volume = volume + vol
            
            if get_squared and volume_sq is None:
                volume_sq = vol**2
            elif get_squared:
                volume_sq = volume_sq + vol**2 
    if get_squared: 
        return volume, volume_sq 
    else:
        return volume

def get_map_offset_volume(image, ref_heights, height_offsets,
                          warped_shift_slopes, inv_inter_camera_map,
                          base_grid=None):
    
    if base_grid is None:
        base_grid = generate_base_grid((image.shape[2], image.shape[3])).cuda()

    # make the grid stack with inter camera shifts
    base_grid = base_grid + inv_inter_camera_map
    base_grid = base_grid.repeat(len(height_offsets), 1, 1, 1)
    ref_heights = ref_heights.repeat(len(height_offsets), 1, 1, 1)

    height_offsets = height_offsets.view(len(height_offsets), 1, 1, 1)
    ref_heights = ref_heights + height_offsets

    slope_shifts = torch.stack([warped_shift_slopes.squeeze(0)] * len(height_offsets))
    slope_shifts = slope_shifts * ref_heights

    grid = base_grid + slope_shifts * -1

    image_stack = torch.stack([image.squeeze(0)] * len(height_offsets), dim=0)

    warped_stack = F.grid_sample(
        image_stack, grid, mode="bilinear", padding_mode="zeros",
        align_corners=False
    )
    return warped_stack