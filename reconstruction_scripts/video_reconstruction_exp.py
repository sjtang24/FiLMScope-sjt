# this script sequntially reconstructs frames from a video 
# re-using the same network for each frame
import time 

from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.recon_util import get_sample_information
from filmscope.config import path_to_data
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


use('Agg') # display in matplotlib

# select sample name and gpu number
sample_name = "knuckle_video"
if torch.cuda.is_available():
    print(f"GPUs Available: {torch.cuda.device_count()} (Using 0)")
    gpu_number = '0' 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("No GPUs available! Not using any GPU's")

# used for neptune logging only
log_description = "shortened knuckle video recon"

# determine frame numbers for this video 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data

input(frame_numbers)
#input(frame_numbers)
dset = None

use_neptune = False

# TODO Choose hyperparameters to loop through
#frame_numbers = np.asarray([420, 421])
parser = argparse.ArgumentParser(description="Run reconstruction with parameters.")
parser.add_argument("--iters", type=int, help="Number of Iterations to run after calibration step")
parser.add_argument("--new_gold_standards", action="store_true", help="use previously run gold standards?")
args = parser.parse_args()

downsample_factors = [1, 2, 4, 8]
camera_arrangements = {
    'all_cameras' : np.arange(0, 48).tolist(),
    'wide_sparse' : [6, 10, 20, 30, 34],
    'narrow_sparse' : [14, 16, 20, 26, 28],
    '2x2 grid' : [20, 21, 26, 27],
    '4x4 grid' : [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34]
}
calib_iterations = 150
iterations = args.iters
display_freq = 150

#time_at_start = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
heightmaps_dir = f"plots/{sample_name}/{args.iters}/heightmap"
losses_dir = f"plots/{sample_name}/{args.iters}/losses"
try:
    os.makedirs(heightmaps_dir)
    os.makedirs(losses_dir)
    os.makedirs("data")
except FileExistsError:
    pass

# INITIALIZATIONS 

#downsample_factors = [downsample]
#times = {key : np.zeros((len(downsample_factors), frame_numbers.size)) for key in camera_arrangements}
#setup_times = {key : np.zeros(len(downsample_factors)) for key in camera_arrangements}
gold_standards = None
if not args.new_gold_standards:
    try:
        with open("data/gold-standards.pkl", 'rb') as gold_standards_file:
            gold_standards = pickle.load(gold_standards_file)
    except Exception as e:
        print("Failed to load existing gold standards:", e)
        gold_standards = None 

is_dict = isinstance(gold_standards, dict)
has_all_frames = is_dict and set(gold_standards.keys()) == set(frame_numbers)
has_all_reconstructions = has_all_frames and all(v is not None for v in gold_standards.values())

if args.new_gold_standards or not has_all_reconstructions:
    print("Gold standards have not all been reconstructed. Restarting...")
    gold_standards = {fm : None for fm in frame_numbers}
else:
    print("Using existing gold standards...")

metrics = {
    cams : {
        ds : {
            fm : {
                'RMSE' : np.zeros(calib_iterations if fm == frame_numbers[0] else iterations), 
                'SSIM' : np.zeros(calib_iterations if fm == frame_numbers[0] else iterations),
                'duration' : np.zeros(calib_iterations if fm == frame_numbers[0] else iterations), 
                'bestimage_RMSE' : None,
                'bestimage_SSIM' : None,
                'best_RMSE' : np.inf,
                'best_RMSE_it' : 0,
                'best_RMSE_time' : 0,
                'best_SSIM' : -np.inf,
                'best_SSIM_it' : 0,
                'best_SSIM_time' : 0
            } for fm in frame_numbers
        } for ds in downsample_factors
    } for cams in camera_arrangements
}
del metrics['all_cameras'][1]

has_all_frame = False
for arr in camera_arrangements:
    for ds in downsample_factors:
        if arr == "all_cameras" and ds == 1:
            continue
        has_all_frame = (has_all_frame or (arr == "all_cameras" and ds == 2)) and (set(metrics[arr][ds].keys()) == set(frame_numbers))
        if not has_all_frame:
            print(f"{arr} {ds}")
print(f"Has all frames: {has_all_frame}")
    
#print(metrics)
# EXPERIMENTAL LOOP
for arrangement_i, arrangement in enumerate(camera_arrangements):
    for downsample_i, downsample in enumerate(downsample_factors):
        if has_all_reconstructions and arrangement == "all_cameras" and downsample == 1:
            print("Skipping all cameras and downsampling factor 1 because we already have gold standards.")
            continue 
        try:
            os.makedirs(f"{heightmaps_dir}/{arrangement}/{downsample}")
            os.makedirs(f"{losses_dir}/{arrangement}/{downsample}")
        except FileExistsError:
            pass
        is_gold_std = arrangement == 'all_cameras' and downsample == 1
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
        if arrangement == 'all_cameras' and downsample == 1:
            metrics[arrangement][downsample] = {'setup_time' : run_manager.setup_time}
        else:
            metrics[arrangement][downsample]['setup_time'] = run_manager.setup_time
        #print(run_manager.setup_time)
        indices = run_manager.dataset.full_crops[20]

        # I assume here that negative indices denote padding so we start with -startx or -starty,
        # and we multiply by the downsample to get the cropping information for the upsampled images 
        # print(f"Downsampling Factor {downsample}")
        startx, endx, starty, endy = tuple([
            -crop_coords if crop_coords < 0 else crop_coords 
            for crop_coords in indices
        ])
        cropping = 64
        sx, sy = startx * downsample + cropping, starty * downsample + cropping
        ex, ey = endx * downsample - cropping, endy * downsample - cropping 
        
        # perform reconstruction
        # if convergence happens quickly for some frames, 
        # follow prompts in the terminal to adjust # iterations used
        # or to manually move on to the next frame
        run_args = config_dict["run_args"]

        for frame_number_i, frame_number in enumerate(frame_numbers):
            if frame_number_i == 0:
                metrics[arrangement][downsample]['final_frames'] = []

            print(f"FRAME #{frame_number} | Downsampling {downsample} | Arrangement {arrangement}")
            #print("")
            #print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
            #print("OR enter 'continue' to move on to the next frame")
            #print("")

            # useful to update description for each frame
            #print(f"Camera Arrangements: {camera_arrangements[arrangement]}")
            run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
            run_manager.swap_frames(frame_number)
            
            losses = []
            #print(f"Starting frame {frame_number}")
            # disable = True removes the progress bar 
            duration = 0
            for i in tqdm(range(run_args["iters"])):
                niter = calib_iterations if frame_number == frame_numbers[0] or (arrangement == "all_cameras" and downsample == 1) else iterations
                done = i == (niter - 1)
                display = i % run_args["display_freq"] == 0
                log = done or display 
                _, _, _, outputs, time_in_ms, loss_values = run_manager.run_epoch(i, log and use_neptune)
                
                duration += time_in_ms
                losses.append(float(loss_values["total"]))
                
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
                heightmap = outputs["depth"].detach().cpu().squeeze().numpy()
                expanded_heightmap = zoom(heightmap, zoom=downsample, order=1)[sx:ex, sy:ey]  # order=1 means linear interpolation   
                reference = run_manager.reference_image.cpu().squeeze()[startx:endx, starty:endy]

                if downsample == 1 and arrangement == 'all_cameras':
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
                
                if done:
                    break
        print(metrics)
        
with open(f'data/metrics-{args.iters}.pkl', 'wb') as f:
    pickle.dump(metrics, f)
    
if not has_all_reconstructions:
    with open('data/gold-standards.pkl', 'wb') as f:
        pickle.dump(gold_standards, f)
    