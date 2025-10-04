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
"""
conda config --add channels conda-forge
conda config --add channels https://conda.anaconda.org/t/ra-cbeb9c0d-37dd-43da-9f38-558244a1fdc0/ramonaoptics
conda config --set channel_priority strict

conda install mamba
mamba install --channel ramonaoptics --yes "python=3.12" "owl-base"

"""

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
CROP_VALUES = (0.125, 0.875, 0.0, 0.9846153846153847)

DEPTH_RANGE = (-16, 48)
N_CAMERAS_X = 8
N_CAMERAS_Y = 6
DOWNSAMPLING = 16
CAMERA_ARRANGEMENT = [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
WARMUP_ITERATIONS = 400
ITERATIONS = 10
CALLIBRATION_FILE = 'data/dice_and_boat/calibration_information'
CROPPING_XE = 1024
CROPPING_YE = 64
CROPPING_XS = 512
CROPPING_YS = 64
FILTER_SIZE = (1, 1)

mcam = MCAM()
mcam.exposure = 150e-3
# specify cameras from array to include
camera_array = np.zeros((N_CAMERAS_X, N_CAMERAS_Y), dtype = bool)
camera_array[2:-2, 1:-1] = True
median_unfold = nn.Unfold(kernel_size = FILTER_SIZE)

timings_dict = {
    'capture' : [],
    'setup' : [],
    'frame_time' : [],
    'post_processing' : []
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
        'height_est' : 0,           
        'crop_size' : (1, 1),       
        'ref_crop_center' : (0.5, 0.5)
    },
    crop_values = CROP_VALUES
)


for i in range(25):
    img_capture_start = time.perf_counter()
    dset = mcam.acquire_selection(camera_array)
    img_capture_end = time.perf_counter()
    timings_dict['capture'].append(img_capture_end - img_capture_start)
    #images_to_consider = dset.images[2:-2, 1:-1, :, :]
   # print(dset)
    setup_start = time.perf_counter()
    if i == 0: #initialize run manager
        run_manager = RunManager(
            timings_dict,
            config_dict, 
            sample = dset, 
            calibration_file=CALLIBRATION_FILE
        )
        timings_dict = run_manager.timing_dict
    else:
        run_manager.swap_frames(sample_image=dset)
        timings_dict = run_manager.timing_dict
    torch.cuda.synchronize()
    setup_end = time.perf_counter()
    timings_dict['setup'].append(setup_end - setup_start)
    
    iters = config_dict["run_args"]["iters"] if i == 0 else ITERATIONS
    losses = []

    if i == 0:
        sx, sy, ex, ey = None, None, None, None
        indices = run_manager.dataset.full_crops[21]

        startx, endx, starty, endy = tuple([
            -crop_coords if crop_coords < 0 else crop_coords
            for crop_coords in indices
        ])

        sx, sy = startx * DOWNSAMPLING + CROPPING_XS, starty * DOWNSAMPLING + CROPPING_YS
        ex, ey = endx * DOWNSAMPLING - CROPPING_XE, endy * DOWNSAMPLING - CROPPING_YE

    total_duration = 0
    for j in tqdm(range(iters)):
        log = (j % config_dict["run_args"]["display_freq"] == 0) or (j == iters - 1)
        epoch_time_start = time.perf_counter()
        mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(
            j, log=(log and config_dict["use_neptune"])
        )
        torch.cuda.synchronize()
        epoch_time_end = time.perf_counter()
        duration = epoch_time_end - epoch_time_start
        total_duration += duration

        if j != iters - 1:
            continue

        timings_dict['frame_time'].append(total_duration)

        losses.append(float(loss_values["total"]))

        fig, (ax0, ax1) = plt.subplots(1, 2, constrained_layout=True) 

        post_start = time.perf_counter()
        depth = outputs["depth"].detach()#.squeeze(0) #.cpu().squeeze()

        #print(f'shape: {depth.shape}')
        depth_upsampled = F.interpolate(
            depth.unsqueeze(0),
            scale_factor = DOWNSAMPLING,
            mode = 'bilinear',
            align_corners = True
        )
        reference_upsampled = F.interpolate(
            run_manager.reference_image.unsqueeze(0),
            scale_factor = DOWNSAMPLING,
            mode = 'bilinear',
            align_corners = True
        )
        #pixel_values = (depth - actual_min) / (actual_max - actual_min + 1e-16)
        #print((actual_min, actual_max))
        #normalized =  pclr.Normalize(vmin = actual_min, vmax = actual_max)
        #depth = depth[128:896, 64:704]
       
        depths_cropped = depth_upsampled[:, :, sy:ey, sx:ex]
        ref_cropped = reference_upsampled[:, :, sy:ey, sx:ex]
        
#        depths = depth_upsampled.squeeze(0).squeeze(0)
        #ref = reference_upsampled.squeeze(0).squeeze(0)

        depth_filtered = median_filter(depths_cropped, FILTER_SIZE, median_unfold)
        torch.cuda.synchronize()
        post_end = time.perf_counter()
        timings_dict['post_processing'].append(post_end - post_start)
        
        depths = depths_cropped.squeeze(0).squeeze(0).cpu().numpy()        
        refs = ref_cropped.squeeze(0).squeeze(0).cpu().numpy()
        
        actual_min, actual_max = depth.min(), depth.max()

        ax0.imshow(refs, cmap='gray')
        depthmap = ax1.imshow(depths, cmap = 'turbo', vmin = actual_min, vmax = actual_max)
        ax0.axis('off')
        ax1.axis('off')

        ax1.set_title(f"Heightmap Reconstruction")
        ax0.set_title(f"Reference Images")
        plt.suptitle(f'Reference and Height Reconstructions for Frame {i} (Iteration {j})\n{datetime.now()}')
        plt.tight_layout()
        cbar = fig.colorbar(
            depthmap, ax = (ax0, ax1), orientation = 'horizontal',
            pad = 0.08
        )
        cbar.set_label('Depth (in mm)')
        
        plt.savefig(f'recons/reconstructions-mcam-frame{i}.png')
        #plt.show()
        """plt.figure()
        plt.plot(run_manager.setup_time_per_frame)
        ax0.set_title(f"Setup Time")
        plt.savefig('setup-mcam.png')

        plt.figure()
        plt.plot(run_manager.times_per_frame)
        ax0.set_title(f"Frame Time")
        plt.savefig('iter-time-mcam.png')"""
run_manager.end()
    
    
"""
print(f"{np.mean(capture_times)} seconds (sdev: {np.std(capture_times)})")
print(dset)
mcam_data.save(dset, "example.nc")"""
with open('timing-results.pkl', 'wb') as timingfile:
    pickle.dump(timings_dict, timingfile)
mcam.close()


# must run sudo python-mcam



"""
frame_numbers = dset.frame_number.data
dset = None

use_neptune = False
config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                   camera_set="all", use_neptune=use_neptune,
                                   log_description=log_description, frame_number=frame_numbers[0],
                                   run_args={"iters": 150, "batch_size": 12, "num_depths": 32,
                                             "display_freq": 1000})
run_manager = RunManager(config_dict)

# perform reconstruction
# if convergence happens quickly for some frames, 
# follow promps in the terminal to adjust # iterations used
# or to manually move on to the next frame
run_args = config_dict["run_args"]
for frame_number in frame_numbers:
    print("")
    print(f"starting for frame {frame_number}")
    print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
    print("OR enter 'continue' to move on to the next frame")
    print("")

    # useful to update description for each frame
    run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
    run_manager.swap_frames(frame_number)
    
    losses = []
    for i in tqdm(range(run_args["iters"])):
        log = (i == run_args["iters"] - 1) or (i % run_args["display_freq"] == 0)
        _, _, _, outputs, loss_values = run_manager.run_epoch(i, log and use_neptune)
        losses.append(float(loss_values["total"]))

        # check here for terminal inputs to move on to next frame if desired
        # or adjust the number of iterations being used
        if select.select([sys.stdin], [], [], 0.1)[0]:
            user_input = sys.stdin.readline().strip()

            command_parts = user_input.split(": ", 1)
            if len(command_parts) == 2:
                command, value = command_parts
                if command.lower() == "iters":
                    print(f"switching to {value} iters on next frame")
                    run_args["iters"] = int(value)
            elif user_input.lower() == "continue":
                print("moving on to next frame...")
                break

        # this block can be edited to save/log in another way
        # this simply displays some outputs with matplotlib
        if log and not use_neptune:
            fig, (ax0, ax1) = plt.subplots(1, 2) 
            ax1.imshow(outputs["depth"].detach().cpu().squeeze(), cmap='turbo')
            ax0.imshow(run_manager.reference_image.cpu().squeeze(), cmap='gray')
            ax1.set_title(f"height map, iteration {i}")
            ax0.set_title(f"reference image, frame {frame_number}")
            plt.show()

            plt.figure()
            plt.plot(losses)
            plt.title(f"losses, frame {frame_number}")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.show()
"""
