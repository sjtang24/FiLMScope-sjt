from owl.instruments import MCAM
import numpy as np
import owl.mcam_data as mcam_data
import time 
import sys
from tqdm import tqdm 
import os
from filmscope.reconstruction import RunManager, generate_config_dict
import matplotlib.pyplot as plt 
import matplotlib.colors as pclr
import torch
import torch.nn as nn
import torch.nn.functional as F
from filmscope.util.filters_pt import median_filter
import pickle
from datetime import datetime
from filmscope.util import load_dictionary, load_image_set, load_from_single_image
from filmscope.recon_util import (tocuda, get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset)


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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CROP_VALUES = (0, 1, 0, 1)
N_ITERATIONS = 100
DEPTH_RANGE = (-5, 5)
N_CAMERAS_X = 8
N_CAMERAS_Y = 6
DOWNSAMPLING = 4
CAMERA_ARRANGEMENT = [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
WARMUP_ITERATIONS = 10
ITERATIONS = 1
CALLIBRATION_FILE = 'data/phantom/calibration_information'
CROPPING_XE = 3056
CROPPING_YE = 64
CROPPING_XS = 3056
CROPPING_YS = 64
#FILTER_SIZE = (1, 5)

mcam = MCAM()
mcam.exposure = 1e-3
# specify cameras from array to include
camera_array = np.zeros((N_CAMERAS_X, N_CAMERAS_Y), dtype = bool)
camera_array[2:-2, 1:-1] = True
#median_unfold = nn.Unfold(kernel_size = FILTER_SIZE)

timings_dict = {
    'capture' : [],
    'setup' : [],
    'frame_time' : [],
    'post_processing' : [],
    'visualization' : []
}

# loop would start here, and can be used in the algorithm 
config_dict = generate_config_dict(
    gpu_number = '0', downsample = DOWNSAMPLING, 
    camera_set = "custom", custom_image_numbers = CAMERA_ARRANGEMENT, 
    run_args = {
        'iters' : WARMUP_ITERATIONS,
        'batch_size' : 12,
        'num_depths' : 32, 
        'display_freq' : 1
    },
    custom_crop_info={
        'depth_range' : DEPTH_RANGE,  
        'height_est' : 5,           
        'crop_size' : (1, 1),       
        'ref_crop_center' : (0.5, 0.5)
    },
    crop_values = CROP_VALUES
)

dset = mcam.acquire_selection(camera_array)
run_manager = RunManager(
    timings_dict,
    config_dict, 
    sample = dset, 
    calibration_file=CALLIBRATION_FILE
)
for i in range(N_ITERATIONS):
    run_manager.run_args["frame_number"] = -1 # trivial
    #run_manager.dataset.to_device("cpu") # time-consuming (0.08)

    frame_number = -1
    sample_image = dset # time-consuming (0.08)

    if frame_number != run_manager.dataset.frame_number or sample_image is not None:
        run_manager.dataset.sample = sample_image
        run_manager.dataset.frame_number = frame_number

        noise = [0, 0]
        torch.cuda.synchronize()
        setup_start = time.perf_counter()
        images = load_image_set(                                   # 0.003s/0.08
            images = run_manager.dataset.sample,
            image_numbers=run_manager.dataset.image_numbers.tolist(),
            downsample=run_manager.dataset.downsample,
            frame_number=run_manager.dataset.frame_number,
            blank_filename=run_manager.dataset.blank_filename
        )

        torch.cuda.synchronize()
        setup_end = time.perf_counter()
        print(setup_end - setup_start)
        images = np.stack([image if len(image.shape) == 3 else image[:, :, None] for image in images], 
                          dtype=np.float32)
        images = torch.from_numpy(images).to("cuda", non_blocking=True)
        noise = torch.rand_like(images) * noise[0] + noise[1]
        images = images + noise 
        images = torch.clamp(images, 0, 255)

        for i, image_number in enumerate(run_manager.dataset.image_numbers.tolist()):
            startx, endx, starty, endy = run_manager.dataset.full_crops[image_number]

            startx2 = max(startx, 0)
            starty2 = max(starty, 0) 
            endx2 = min(endx, images.shape[1] - 1) 
            endy2 = min(endy, images.shape[2] - 1)

            if endx2 != endx:
                endx = endx2 - endx
            else:
                endx = endx - startx
            if endy2 != endy:
                endy = endy2 - endy
            else:
                endy = endy - starty

            run_manager.dataset.images[i, :, startx2 - startx:endx, starty2 - starty:endy] = (
                images[i, startx2:endx2, starty2:endy2].permute([2, 0, 1]))
   
    run_manager.reference_image = run_manager.dataset.reference_image.cuda() # trivial (~0)


    """TIMING"""   
    dataset = run_manager.dataset
    depth_values = run_manager.depth_values
    get_squared = True
    volume = None
    volume_sq = None 
    sample_cuda = tocuda(dataset.get_full_sample())
    images = sample_cuda["imgs"]
    warped_ss_maps = sample_cuda["warped_shift_slope_maps"] - torch.asarray(dataset.ref_camera_shift_slopes).cuda()
    iic_maps = sample_cuda["inv_inter_camera_maps"]

    print(images.shape)
    print(iic_maps.shape)
    print(warped_ss_maps.shape)
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
    """for image, iic_map, warped_ss_map in zip(images, iic_maps, warped_ss_maps):

        

        # trivial 
        warped_volume = warped_volume.permute(1, 0, 2, 3)[None]
        if volume is None:
            volume = warped_volume
        else:
            volume = volume + warped_volume

        if get_squared and volume_sq is None:
            volume_sq = warped_volume**2
        elif get_squared:
            volume_sq = volume_sq + warped_volume**2"""
        
    num_views = len(run_manager.dataset)
    run_manager.volume_variance = volume_sq.div_(num_views).sub_(
        volume.div_(num_views).pow_(2)
    )

    #run_manager.dataset.to_device("cuda") # 0.02-0.03


run_manager.end()
mcam.close()
