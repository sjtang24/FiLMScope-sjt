# this script sequntially reconstructs frames from a video 
# re-using the same network for each frame
import time 

from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.recon_util import get_sample_information
from filmscope.config import path_to_data

import xarray as xr
import os
from tqdm import tqdm
import sys 
import select
from matplotlib import use
from matplotlib import pyplot as plt
import numpy as np
import torch 

use('Agg') # display in matplotlib
print(torch.cuda.device_count())
# select sample name and gpu number
sample_name = "knuckle_video"
gpu_number = '0' 
if torch.cuda.is_available():
    print(f"GPUs Available: {torch.cuda.device_count()} (Using 1)")
    gpu_number = '1' 
else:
    print("No GPUs available!")
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# used for neptune logging only
log_description = "shortened kncukle video recon"

# determine frame numbers for this video 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data

dset = None

use_neptune = False

try:
    os.makedirs(f"plots/heightmap/{sample_name}")
    os.makedirs(f"plots/losses/{sample_name}")
    os.makedirs(f"plots/timing/{sample_name}")
    os.makedirs(f"timings")
except FileExistsError:
    pass

downsample_factors = [2, 4, 7, 8, 16, 32]
#downsample_factors = [downsample]
times = np.zeros((len(downsample_factors), frame_numbers.size))

for downsample_i, downsample in enumerate(downsample_factors):
    # print(f"Timing downsampling factor of {downsample}...")
    config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=downsample,
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
    #print(run_args)
    for frame_number_i, frame_number in enumerate(frame_numbers):
        #print("")
        #print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
        #print("OR enter 'continue' to move on to the next frame")
        #print("")

        # useful to update description for each frame
        run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
        run_manager.swap_frames(frame_number)
        
        losses = []
        print(f"Starting frame {frame_number}")
        start_time = time.time()
        for i in tqdm(range(run_args["iters"]), disable = True):
            log = (i == run_args["iters"] - 1) # or (i > 0 and i % run_args["display_freq"] == 0)
            _, _, _, outputs, loss_values = run_manager.run_epoch(i, log and use_neptune)
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
            if log and not use_neptune:
                frame_duration = time.time() - start_time
                print(f"Finished frame {frame_number}")
                times[downsample_i, frame_number_i] = frame_duration
                
                fig, (ax0, ax1) = plt.subplots(1, 2) 
                ax1.imshow(outputs["depth"].detach().cpu().squeeze(), cmap='turbo')
                ax0.imshow(run_manager.reference_image.cpu().squeeze(), cmap='gray')
                ax1.set_title(f"height map, iteration {i}")
                ax0.set_title(f"reference image, frame {frame_number}\n(Downsampling = {downsample})")
                plt.savefig(f"plots/heightmap/{sample_name}/{frame_number}_ds{downsample}.png")
                #plt.show()

                plt.figure()
                plt.plot(losses)
                plt.title(f"losses, frame {frame_number}, downsampling {downsample}")
                plt.xlabel("iteration")
                plt.ylabel("loss")
                plt.savefig(f"plots/losses/{sample_name}/{frame_number}_ds{downsample}.png")
                #plt.show()
        # print(f"Execution runtime for frame: {frame_duration}s")
    #print(f"Total runtime for downsample factor {downsample}: {np.sum(times[downsample_i, :])}s")
    #print(f"Runtime/Frame for downsample factor {downsample}: {np.mean(times[downsample_i, :])}s +/- {np.std(times[downsample_i, :])}")

np.save(f"timings/video-reconstruction_{sample_name}.npy", times)