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
import numpy as np
import torch 
from datetime import datetime
import pickle
import argparse

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
parser.add_argument("--sample_name", type=str, help = "name of the sample", default = "knuckle_video")
parser.add_argument("--save_iters", action="store_true", help="save every iteration")
parser.add_argument("--save_final", action="store_true", help="save last iteration")
args = parser.parse_args()

## select sample name and gpu number
sample_name = args.sample_name
gpu_number = ""

# check if script running on correct sample 
if args.sample_name in ["knuckle_video"]:
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
dset = xr.open_dataset(alt_path if arg.sample_name == "skull_tool_video" else path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data
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
    downsample_factors = [1]
    camera_arrangements = {
        'all_cameras' : np.arange(0, 48).tolist(),
        'wide_sparse' : [6, 10, 20, 30, 34],
        'narrow_sparse' : [14, 16, 20, 26, 28],
        '2x2 grid' : [20, 21, 26, 27],
        '4x4 grid' : [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
    }
    iterations = args.iters
else:
    print("Generating gold standards (with all cameras and no downsampling). Ignoring iterations argument")
    downsample_factors = [1]
    camera_arrangements = {
        'all_cameras' : np.arange(0, 48).tolist()
    }

#time_at_start = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')

# INITIALIZATIONS 
#downsample_factors = [downsample]
#times = {key : np.zeros((len(downsample_factors), frame_numbers.size)) for key in camera_arrangements}
#setup_times = {key : np.zeros(len(downsample_factors)) for key in camera_arrangements}
# gold_standards = None
# if not args.new_gold_standards:
#     try:
#         with open("data/gold-standards.pkl", 'rb') as gold_standards_file:
#             gold_standards = pickle.load(gold_standards_file)
#     except Exception as e:
#         print("Failed to load existing gold standards:", e)
#         gold_standards = None 

# is_dict = isinstance(gold_standards, dict)
# has_all_frames = is_dict and set(gold_standards.keys()) == set(frame_numbers)
# has_all_reconstructions = has_all_frames and all(v is not None for v in gold_standards.values())

# if args.new_gold_standards or not has_all_reconstructions:
#     print("Gold standards have not all been reconstructed. Restarting...")
#     gold_standards = {fm : None for fm in frame_numbers}
# else:
#     print("Using existing gold standards...")
 
#print(metrics)

startx, starty, endx, endy = None, None, None, None
sx, ex = None, None 
sy, ey = None, None 
cropping = 64 
iteration_data = None 

real_order_frames = np.argsort(frame_numbers)

# EXPERIMENTAL LOOP
for downsample_i, downsample in enumerate(downsample_factors):
    for arrangement_i, arrangement in enumerate(camera_arrangements):
        config_dict = \
            generate_config_dict(
                sample_name=sample_name, gpu_number=gpu_number, downsample=downsample,
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
        print(run_manager.setup_time)
        
        # perform reconstruction
        # if convergence happens quickly for some frames, 
        # follow prompts in the terminal to adjust # iterations used
        # or to manually move on to the next frame
        run_args = config_dict["run_args"]
        
        if arrangement_i == 0 and downsample_i == 0:
            indices = run_manager.dataset.full_crops[20]

            # I assume here that negative indices denote padding so we start with -startx or -starty,
            # and we multiply by the downsample to get the cropping information for the upsampled images 
            # print(f"Downsampling Factor {downsample}")
            startx, endx, starty, endy = tuple([
                -crop_coords if crop_coords < 0 else crop_coords 
                for crop_coords in indices
            ])

            sx, sy = startx * downsample + cropping, starty * downsample + cropping
            ex, ey = endx * downsample - cropping, endy * downsample - cropping 
            
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
            frame_number = actual_frame
            if frame_number < 450:
                continue 
            print(f"Downsampling {downsample} | Arrangement {arrangement} || FRAME #{frame_number}")
            #print("")
            #print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
            #print("OR enter 'continue' to move on to the next frame")
            #print("")

            # useful to update description for each frame
            #print(f"Camera Arrangements: {camera_arrangements[arrangement]}")
            run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
            run_manager.swap_frames(frame_number)
            
            niter = calib_iterations if frame_number == frame_numbers[0] or (arrangement == "all_cameras" and downsample == 1 and args.only_gold_standards) else iterations
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
                
                duration += time_in_ms                
                heightmap = outputs["depth"].detach().cpu().squeeze().numpy()
                expanded_heightmap = zoom(heightmap, zoom=downsample, order=1)[sx:ex, sy:ey]  # order=1 means linear interpolation   
                reference = run_manager.reference_image.cpu().squeeze()[sx:ex, sy:ey]
                config_data['reference'] = reference
                iter_reconstruct[i, :, :] = expanded_heightmap
                config_data['duration'][i] = duration / 1000
                config_data['losses'][i] = float(loss_values["total"])

                # save periodically (every 10 frame for gold standards, or after all data for non-gold-standards)
            
                # check here for terminal inputs to move on to next frame if desired
                # or adjust the number of iterations being used
                # if select.select([sys.stdin], [], [], 0.1)[0]:
                #     user_input = sys.stdin.readline().strip()

                #     command_parts = user_input.split(": ", 1)
                #     if len(command_parts) == 2:
                #         command, value = command_parts
                #         if command.lower() == "iters":
                #             print(f"switching to {value} iters on next frame")
                #             run_args["iters"] = int(value)
                #     elif user_input.lower() == "continue":
                #         print("moving on to next frame...")
                #         break

                # this block can be edited to save/log in another way
                # this simply displays some outputs with matplotlib
            recon_frames[frame_number_i, :, :] = iter_reconstruct[-1, :, :]
            if args.save_iters:
                if not args.only_gold_standards:
                    datapath = reconpath + f'/{args.iters}/{arrangement}/{downsample}'
                else:
                    datapath = reconpath + "goldstandard"
                try: 
                    os.makedirs(datapath)
                except FileExistsError:
                    pass
                np.save(f'{datapath}/{frame_number}.npy', iter_reconstruct)
            frame_number_i += 1
        # frames loop ends
        if args.save_final:
            if not args.only_gold_standards:
                datapath = reconpath + f'{args.iters}_{"-".join(arrangement.split(" "))}_{downsample}.npy'
            else:
                datapath = reconpath + 'goldstandard.npy'
            np.save(datapath, recon_frames) 
    # downsampling loop ends
# camera arrangement loop ends

save_pickle(iteration_data, args)
# Final File Upload




"""if downsample == 1 and arrangement == 'all_cameras':
if done:
print("Storing gold standard!")
gold_standards[frame_number] = heightmap[sx:ex, sy:ey]
else:
metrics_for_iter = metrics[arrangement][downsample]
if done:    
metrics_for_iter['final_frames'].append(expanded_heightmap)
curr_metrics = metrics_for_iter[frame_number]
rmse = float(np.sqrt(
np.mean(
np.square(
    expanded_heightmap - gold_standards[frame_number]
)
)
))        
mssim = structural_similarity(
expanded_heightmap, gold_standards[frame_number], full=False
)

curr_metrics['RMSE'][i] = rmse
curr_metrics['SSIM'][i] = mssim
curr_metrics['duration'][i] = duration / 1000
if curr_metrics['best_RMSE'] > rmse:
curr_metrics['best_RMSE'] = rmse
curr_metrics['bestimage_RMSE'] = expanded_heightmap
curr_metrics['best_RMSE_it'] = i
curr_metrics['best_RMSE_time'] = curr_metrics['duration'][i]
if curr_metrics['best_SSIM'] < mssim:
curr_metrics['best_SSIM'] = mssim
curr_metrics['bestimage_SSIM'] = expanded_heightmap
curr_metrics['best_SSIM_it'] = i
curr_metrics['best_SSIM_time'] = curr_metrics['duration'][i]

if log and not use_neptune:                
fig = plt.figure()
width_ratios = [1, 1, 1]
gs = GridSpec(1, 3, width_ratios=width_ratios)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
ax0.axis('off')
ax1.axis('off')
ax2.axis('off')

ax2.imshow(expanded_heightmap, cmap = 'turbo')
if gold_standards is not None and gold_standards[frame_number] is not None:
ax1.imshow(gold_standards[frame_number], cmap='turbo')
ax0.imshow(reference, cmap='gray')
ax2.set_title(f"Reconstructed")
ax1.set_title(f"Gold Standard")
ax0.set_title(f"Reference Image")
plt.savefig(f"{heightmaps_dir}/{arrangement}/{downsample}/{frame_number}.png")
plt.close('all')

plt.figure()
plt.plot(losses)
plt.title(f"losses, frame {frame_number}, downsampling {downsample}, {arrangement}")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig(f"{losses_dir}/{arrangement}/{downsample}/{frame_number}.png")
plt.close('all')

print(metrics)


if not has_all_reconstructions:
with open('data/gold-standards.pkl', 'wb') as f:
pickle.dump(gold_standards, f)
"""