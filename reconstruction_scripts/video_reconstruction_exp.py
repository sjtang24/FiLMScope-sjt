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
import time 
import pickle

use('Agg') # display in matplotlib

print(torch.cuda.device_count())
# select sample name and gpu number
sample_name = "knuckle_video"
gpu_number = '0' 
if torch.cuda.is_available():
    print(f"GPUs Available: {torch.cuda.device_count()} (Using 2)")
    gpu_number = '1' 
else:
    print("No GPUs available!")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# used for neptune logging only
log_description = "shortened knuckle video recon"

# determine frame numbers for this video 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data

dset = None

use_neptune = False

time_at_start = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
try:
    os.makedirs(f"plots/{time_at_start}/heightmap/{sample_name}")
    os.makedirs(f"plots/{time_at_start}/losses/{sample_name}")
    os.makedirs(f"plots/{time_at_start}/timing/{sample_name}")
    os.makedirs(f"timings/{time_at_start}")
except FileExistsError:
    pass

# TODO Choose hyperparameters to loop through
frame_numbers = np.asarray([420])
downsample_factors = [1, 2, 4, 8, 16, 32]

#downsample_factors = [downsample]
times = np.zeros((len(downsample_factors), frame_numbers.size))
setup_times = np.zeros(len(downsample_factors))
gold_standards = None
RMSE = None

for downsample_i, downsample in enumerate(downsample_factors):
    # print(f"Timing downsampling factor of {downsample}...")
    config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=downsample,
                                    camera_set="all", use_neptune=use_neptune,
                                    log_description=log_description, frame_number=frame_numbers[0],
                                    run_args={"iters": 150, "batch_size": 12, "num_depths": 32,
                                                "display_freq": 25})
    run_manager = RunManager(config_dict)
    setup_times[downsample_i] = run_manager.setup_time
    print(run_manager.setup_time)
    indices = run_manager.dataset.full_crops[20]

    # I assume here that negative indices denote padding so we start with -startx or -starty,
    # and we multiply by the downsample to get the cropping information for the upsampled images 
    # print(f"Downsampling Factor {downsample}")
    startx, endx, starty, endy = tuple([
        -crop_coords if crop_coords < 0 else crop_coords 
        for crop_coords in indices
    ])
    sx, sy = startx * downsample, starty * downsample
    ex, ey = endx * downsample, endy * downsample
    
    # perform reconstruction
    # if convergence happens quickly for some frames, 
    # follow prompts in the terminal to adjust # iterations used
    # or to manually move on to the next frame
    run_args = config_dict["run_args"]
    if downsample == 1: 
        gold_standards = {fn : None for fn in frame_numbers}
        RMSE = {ds : [] for ds in downsample_factors if ds > 1}
        SSIM = {ds : [] for ds in downsample_factors if ds > 1}
    
    for frame_number_i, frame_number in enumerate(frame_numbers):
        #print("")
        #print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
        #print("OR enter 'continue' to move on to the next frame")
        #print("")

        # useful to update description for each frame
        run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
        run_manager.swap_frames(frame_number)
        
        losses = []
        #print(f"Starting frame {frame_number}")
        # disable = True removes the progress bar 
        duration = 0
        for i in tqdm(range(run_args["iters"]), disable = downsample > 1):
            done = i == run_args["iters"] - 1
            display = i > 0 and i % run_args["display_freq"] == 0
            log = done or display 

            start_time = time.perf_counter()
            _, _, _, outputs, loss_values = run_manager.run_epoch(i, log and use_neptune)
            duration += (time.perf_counter() - start_time)
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

            if done:
                if downsample == 1: 
                    gold_standards[frame_number] = heightmap
                times[downsample_i, frame_number_i] = duration

            # TODO change this to linear interpolation
            #expanded_heightmap = heightmap.repeat_interleave(downsample, dim = 0).repeat_interleave(downsample, dim = 1)[
            #    sx:ex, sy:ey
            #]
            expanded_heightmap = zoom(heightmap, zoom=downsample, order=1)[sx:ex, sy:ey]  # order=1 means linear interpolation   
            reference = run_manager.reference_image.cpu().squeeze()[startx:endx, starty:endy]

            if downsample > 1:
                RMSE[downsample].append(
                    int(np.sqrt(
                        np.sum(
                            np.square(
                                expanded_heightmap - gold_standards[frame_number]
                            )
                        )
                    ))
                )
                mssim = structural_similarity(
                    expanded_heightmap, gold_standards[frame_number], full=False
                )
                SSIM[downsample].append(mssim)

            if log and not use_neptune and downsample > 1:                
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
                ax1.imshow(gold_standards[frame_number], cmap='turbo')
                ax0.imshow(reference, cmap='gray')
                ax2.set_title(f"Reconstructed")
                ax1.set_title(f"Gold Standard")
                ax0.set_title(f"Reference Image")
                plt.savefig(f"plots/{time_at_start}/heightmap/{sample_name}/{frame_number}_ds{downsample}.png")
                #plt.show()

                plt.figure()
                plt.plot(losses)
                plt.title(f"losses, frame {frame_number}, downsampling {downsample}")
                plt.xlabel("iteration")
                plt.ylabel("loss")
                plt.savefig(f"plots/{time_at_start}/losses/{sample_name}/{frame_number}_ds{downsample}.png")
                #plt.show()

        # print(f"Execution runtime for frame: {frame_duration}s")
    #print(f"Total runtime for downsample factor {downsample}: {np.sum(times[downsample_i, :])}s")
    #print(f"Runtime/Frame for downsample factor {downsample}: {np.mean(times[downsample_i, :])}s +/- {np.std(times[downsample_i, :])}")
    print(RMSE)
np.save(f"timings/video-reconstruction_{sample_name}.npy", times)
np.save(f"timings/setup_{sample_name}.npy", setup_times)

with open('rmses.pkl', 'wb') as f:
    pickle.dump(RMSE, f)
with open('ssim.pkl', 'wb') as f:
    pickle.dump(SSIM, f)   