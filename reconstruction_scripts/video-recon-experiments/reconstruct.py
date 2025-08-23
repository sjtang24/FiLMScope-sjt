"""
~/FiLMScope/reconstruction_scripts/video-recon-experiments/reconstruct.py

Description:
    This computes 3D reconstructions each frame of the video associated with SAMPLE_NAME,
    over various configurations, based on downsampling factors and camera configurations.
        - Downsampling Factors: [1, 2, 4, 8]
        - Cameras Configuration: ['all_cameras', 'wide_sparse', 'narrow_sparse', '2x2 grid', '4x4 grid]

    If the --only_gold_standards flag is used, it compute the gold standard reconstructions.
    Otherwise, for each paired configuration, after the "warmup" on the first frame, each
    subsequent frame runs for ITERS iterations. 

    The program saves reconstructions across iterations as pickle files into data [ITERS, H, W]

Usage:
    reconstruct.py [-h] [--iters ITERS] [--only_gold_standards] [--gpu GPU] [--sample_name SAMPLE_NAME]

    Run reconstruction with parameters.

    optional arguments:
    -h, --help                      show this help message and exit
    --iters ITERS                   Number of Iterations to run after calibration step
    --only_gold_standards           generate only gold standards
    --gpu GPU                       gpu number (0-4, incl.)
    --sample_name SAMPLE_NAME       name of the sample

Author: Steven Tang
Date: 2025-07-24
"""

import time 

from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.recon_util import get_sample_information
from filmscope.config import path_to_data, alt_path
from skimage.metrics import structural_similarity
from scipy.ndimage import zoom


import xarray as xr
import os
from tqdm import tqdm
import sys 
import select
from matplotlib import use
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import torch.nn.functional as F
import numpy as np
import torch 
import torch.nn as nn 
from datetime import datetime
import pickle
import argparse
#from scipy.ndimage import gaussian_filter, median_filter
from filters_pt import median_filter
import copy

# CONFIGURATION INFORMATION FOR VIDEO DATASETS
DOWNSAMPLING_MULTIPLIER = {
    'skull_tool_video' : 1,
    'knuckle_video' : 1,
    'eye_video' : 3
}

CROPPING_FACTOR = {
    'skull_tool_video' : 8,
    'knuckle_video' : 64,
    'eye_video' : 64
}

FILTER = {
    'median' : (median_filter, (5, 5), nn.Unfold(kernel_size=(5, 5))),
    'none' : (None, None, None)
}

camera_arrangements = {
    'all_cameras' : np.arange(0, 48).tolist(),                                     
    'wide_sparse' : [6, 10, 20, 30, 34],                                          
    'narrow_sparse' : [14, 16, 20, 26, 28],                                        
    '2x2 grid' : [20, 21, 26, 27],                                                
    '4x4 grid' : [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34] 
}

camera_arrangements_gold = {
    'all_cameras' : np.arange(0, 48).tolist()
}

CAMERA_ARRANGEMENTS = {
    'skull_tool_video' : copy.deepcopy(camera_arrangements),
    'knuckle_video' : copy.deepcopy(camera_arrangements),
    'eye_video' : copy.deepcopy(camera_arrangements)
}
del CAMERA_ARRANGEMENTS['eye_video']['all_cameras']
CAMERA_ARRANGEMENTS['eye_video']['wide_sparse'] = [13, 16, 21, 31, 34] 
CAMERA_ARRANGEMENTS['eye_video']['narrow_sparse'] = [14, 16, 21, 26, 28]
#input(CAMERA_ARRANGEMENTS)

CAMERA_ARRANGEMENTS_GOLD = {
    'skull_tool_video' : {'all_cameras' : camera_arrangements['all_cameras']},
    'knuckle_video' : {'all_cameras' : camera_arrangements['all_cameras']}, 
    'eye_video' : {'4x4 grid' : camera_arrangements['4x4 grid']}
}

REF_IMG = {
    'skull_tool_video' : 20,
    'knuckle_video' : 20,
    'eye_video' : 21
}

def save_pickle(data, args):
    if args.only_gold_standards:
        with open(f'data/iter-gold-standards.pkl', 'wb') as f:
            pickle.dump(data, f)
    else:
        with open(f'data/iter-{args.iters}.pkl', 'wb') as f:
            pickle.dump(data, f)

reconpath = 'recon/'
try: 
    os.makedirs('data')
except FileExistsError:
    pass

## expect parameters 
parser = argparse.ArgumentParser(description="Run reconstruction with parameters.")
parser.add_argument("--iters", type=int, help="Number of Iterations to run after calibration step")
parser.add_argument("--only_gold_standards", action="store_true", help="generate only gold standards")
parser.add_argument("--gpu", type=int, help="gpu number (0-4, incl.)", default = -1)
parser.add_argument("--sample_name", type=str, help = "name of the sample", default = "skull_tool_video")
parser.add_argument("--save_iters", action="store_true", help="save every iteration")
parser.add_argument("--save_final", action="store_true", help="save last iteration")
parser.add_argument("--frame_end", type=int, help="Frame to end reconstruction (0...MAX_FRAMES)")
parser.add_argument("--frame_start", type=int, help="Frame to start reconstructions (0, ..., MAX_FRAMES - 1)")
parser.add_argument("--filter", type=str, help = "Type of Filter ('gaussian', 'median', or 'none')", default = 'none')
parser.add_argument("--every10", action="store_true", help = "Skip 9 frames so we process frames 0, 10, 20, ..., n")
args = parser.parse_args()

## select sample name and gpu number
sample_name = args.sample_name
gpu_number = ""

# check if script running on correct sample 
if args.sample_name in ["knuckle_video", "skull_tool_video", "eye_video"]:
    sample_name = args.sample_name
else:
    print("Sample does not exist!")
    sys.exit()

# check if gpu number is valid 
if 0 <= args.gpu and args.gpu <= torch.cuda.device_count():
    print(f"All GPUs available! Using GPU {args.gpu}...")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    gpu_number = str(args.gpu)
else:
    print("No GPUs available! Not using any GPU's")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# determine frame numbers for this video 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + image_filename)
frame_numbers = dset.frame_number.data

if args.frame_start is None:
    args.frame_start = 0
if args.frame_end is None:
    args.frame_end = frame_numbers[-1]

MIN_FRAMES_NUMBER = np.min(frame_numbers)
MAX_FRAMES_NUMBER = np.max(frame_numbers)
if not (MIN_FRAMES_NUMBER <= args.frame_start < args.frame_end <= MAX_FRAMES_NUMBER):
    print(f"Bad frame start and end parameters! Must be between {MIN_FRAMES_NUMBER} and {MAX_FRAMES_NUMBER}")
    sys.exit()

print(f"Viewing Frame #{args.frame_start} to Frame #{args.frame_end}")

filter_func, filter_width, filter_unfold = None, None, None  
if args.filter not in ['gaussian', 'median', 'none']:
    print("Filter is invalid! Must be Gaussian or Median Filter or 'none.'")
elif args.filter not in FILTER:
    print("Filter specified is not implemented!")
    sys.exit()
else:
    filter_func, filter_width, filter_unfold = FILTER[args.filter]
    print(f"Using {args.filter.upper()} filtering with width (standard deviation or size) {filter_width}")

dset = None
use_neptune = False
log_description = ""

calib_iterations = 150
iterations = calib_iterations
display_freq = calib_iterations
directory_names = None
camera_arrangements = None 
downsample_factors = None 

if (args.iters is not None) == args.only_gold_standards:
    print("Quitting because of bad inputs! Either both or neither of the arguments were provided. Use one or the other.")
    sys.exit()

if args.save_iters == args.save_final:
    print("Either save all or the final iteration.")
    sys.exit()

if not args.only_gold_standards:
    downsample_factors = [8, 4, 2, 1]
    camera_arrangements = CAMERA_ARRANGEMENTS[args.sample_name]
    iterations = args.iters
else:
    print("Generating gold standards (with all cameras and no downsampling). Ignoring iterations argument")
    downsample_factors = [1]
    camera_arrangements = CAMERA_ARRANGEMENTS_GOLD[args.sample_name]
#input(camera_arrangements)

startx, starty, endx, endy = None, None, None, None
sx, ex = None, None 
sy, ey = None, None 
cropping = CROPPING_FACTOR[sample_name] # 64 = knuckle video, 4 = skull_tool_video 
iteration_data = None 

real_order_frames = np.argsort(frame_numbers)
#print(real_order_frames)
# EXPERIMENTAL LOOP
for downsample_i, downsample in enumerate(downsample_factors):
    for arrangement_i, arrangement in enumerate(camera_arrangements):
        config_dict = \
            generate_config_dict(
                sample_name=sample_name, gpu_number=gpu_number, downsample=downsample * DOWNSAMPLING_MULTIPLIER[sample_name],
                camera_set="custom", use_neptune=use_neptune,
                custom_image_numbers=camera_arrangements[arrangement],
                log_description=log_description, frame_number=frame_numbers[0],
                run_args= {
                    "iters": calib_iterations, 
                    "batch_size": 12, 
                    "num_depths": 32,
                    "display_freq": display_freq
                }
            )
        run_manager = RunManager(config_dict)
        
        # perform reconstruction
        # if convergence happens quickly for some frames, 
        # follow prompts in the terminal to adjust # iterations used
        # or to manually move on to the next frame
        run_args = config_dict["run_args"]
        
        if arrangement_i == 0 and downsample_i == 0:
            indices = run_manager.dataset.full_crops[REF_IMG[sample_name]]

            # I assume here that negative indices denote padding so we start with -startx or -starty,
            # and we multiply by the downsample to get the cropping information for the upsampled images 
            # print(f"Downsampling Factor {downsample}")
            startx, endx, starty, endy = tuple([
                -crop_coords if crop_coords < 0 else crop_coords 
                for crop_coords in indices
            ])

            sx, sy = startx * downsample * DOWNSAMPLING_MULTIPLIER[sample_name] + cropping, starty * downsample * DOWNSAMPLING_MULTIPLIER[sample_name] + cropping
            ex, ey = endx * downsample * DOWNSAMPLING_MULTIPLIER[sample_name] - cropping, endy * downsample * DOWNSAMPLING_MULTIPLIER[sample_name] - cropping 
            
            leny = ey - sy
            lenx = ex - sx
            iteration_data = {
                ds : {
                    cams : {
                            fm : {
                                'reference' : None
                            } for fm in frame_numbers
                    } for cams in camera_arrangements
                } for ds in downsample_factors
            }

        print(f"Downsampling {downsample} | Arrangement {arrangement}")
        iteration_data[downsample][arrangement]['setup_time'] = run_manager.setup_time
        print(f"Setup: {iteration_data[downsample][arrangement]['setup_time']}")
        num_frames_actual = frame_numbers[real_order_frames][-1] + 1
        recon_frames = np.zeros((num_frames_actual, lenx, leny))  
        frame_number_i = 0 
        prev_actual_frame = None 
        for fnum in tqdm(real_order_frames):
            actual_frame = frame_numbers[fnum]
            if prev_actual_frame is not None and prev_actual_frame == actual_frame:
                continue # skip repeated frame
            prev_actual_frame = actual_frame
            if actual_frame < args.frame_start:
                continue # skip frames before start
            if actual_frame > args.frame_end:
                break # stop it for this configuration
            if args.every10 and actual_frame % 10 != 0:
                continue 
            frame_number = actual_frame
        
            print(f"Downsampling {downsample} | Arrangement {arrangement} || FRAME #{frame_number}")
            #print("")
            #print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
            #print("OR enter 'continue' to move on to the next frame")
            #print("")

            # useful to update description for each frame
            #print(f"Camera Arrangements: {camera_arrangements[arrangement]}")
            run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
            run_manager.swap_frames(frame_number)
            
            niter = calib_iterations if frame_number == args.frame_start or (arrangement == "all_cameras" and downsample == 1 and args.only_gold_standards) else iterations
            #print(f"Starting frame {frame_number}")
            # disable = True removes the progress bar 
            duration = 0
            config_data = iteration_data[downsample][arrangement][frame_number]
            config_data['losses'] = np.zeros(niter)
            config_data['duration'] = np.zeros(niter)
            iter_reconstruct = np.zeros((niter, lenx, leny))
            for i in tqdm(range(niter)):
                done = i == (niter - 1)
                display = i % run_args["display_freq"] == 0
                log = done or display 
                _, _, _, outputs, time_in_ms, loss_values = run_manager.run_epoch(i, log and use_neptune)
                
                duration += time_in_ms # in ms

                #print(f'{sx}:{ex} | {sy}:{ey}')

                # time how long it takes to postprocessing steps, including upsampling and filtering 
                start_post_proc_timer = time.perf_counter()             
                heightmap = outputs["depth"].detach().contiguous()
                heightmap_upsampled = F.interpolate(
                    heightmap.unsqueeze(0), #.cpu().squeeze().numpy(),
                    scale_factor=downsample * DOWNSAMPLING_MULTIPLIER[sample_name], 
                    mode='bilinear', 
                    align_corners=True
                ) #.squeeze().squeeze() # H x W
                reconstruction = heightmap_upsampled[:, :, sx:ex, sy:ey]  # order=1 means linear interpolation 
                """reconstruction = zoom(
                    heightmap, zoom=downsample * DOWNSAMPLING_MULTIPLIER[sample_name], order=1
                )"""

               # print(reconstruction.shape)

                if filter_func is not None:
                    reconstruction = filter_func(reconstruction, filter_width, filter_unfold)
                torch.cuda.synchronize()
                end_post_proc_timer = time.perf_counter()  # in seconds
                duration += (1000 * (end_post_proc_timer - start_post_proc_timer))
                # end timing 
                reconstruction = reconstruction.squeeze(0).squeeze(0).cpu().numpy()
                iter_reconstruct[i, :, :] = reconstruction 
                config_data['duration'][i] = duration / 1000
                config_data['losses'][i] = float(loss_values["total"])
                
            
                # save information 

            reference = run_manager.reference_image.cpu().squeeze()[sx:ex, sy:ey]
            config_data['reference'] = reference
            recon_frames[frame_number_i, :, :] = iter_reconstruct[-1, :, :]

            """recon = recon_frames[frame_number_i, :, :]
            n_recon = (recon - recon.min()) / (recon.max() - recon.min())
                
            plt.figure()
            plt.imshow((reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min()), 
                        cmap = "turbo", vmin = 0, vmax = 1)
            plt.axis('off')     # hides axes for this subplot
            plt.savefig('recon.png')"""
            frame_number_i += 1
            #print(f"Frame #{frame_number_i}")
            #input(recon_frames[frame_number_i, :, :])
            if args.save_iters:
                if not args.only_gold_standards:
                    datapath = reconpath + sample_name + f'/{args.iters}/{arrangement}/{downsample}'
                else:
                    datapath = reconpath + sample_name + "/goldstandard"
                try: 
                    os.makedirs(datapath + sample_name)
                except FileExistsError:
                    pass
                np.save(f'{datapath}/{frame_number}.npy', iter_reconstruct)
        #print(recon_frames)
        # frames loop ends
        if args.save_final:
            if not args.only_gold_standards:
                datapath = reconpath + sample_name + f'/{args.iters}_{"-".join(arrangement.split(" "))}_{downsample}.npy'
            else:
                datapath = reconpath + sample_name + '/goldstandard.npy'
            try: 
                os.makedirs(reconpath + sample_name)
            except FileExistsError:
                pass
            #input(recon_frames[args.frame_start:args.frame_end, :, :].shape)
            #input(recon_frames[args.frame_start:args.frame_end, :, :])
            np.save(datapath, recon_frames[args.frame_start:args.frame_end, :, :]) 
    # downsampling loop ends
# camera arrangement loop ends

save_pickle(iteration_data, args)
# Final File Upload
