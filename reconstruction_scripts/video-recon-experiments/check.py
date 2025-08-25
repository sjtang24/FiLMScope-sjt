# this script sequntially reconstructs frames from a video 
# re-using the same network for each frame

from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.recon_util import get_sample_information
from filmscope.config import path_to_data, alt_path

import xarray as xr
import os
from tqdm import tqdm
import sys 
import select
from matplotlib import pyplot as plt

# select sample name and gpu number
sample_name = "skull_tool_video"
gpu_number = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# used for neptune logging only
log_description = ""

# determine frame numbers for this video 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(alt_path + '/' + image_filename)
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

    # useful to update description for each frame
    run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
    run_manager.swap_frames(frame_number)
    
    losses = []
    for i in tqdm(range(run_args["iters"])):
        _, _, _, outputs, _, loss_values = run_manager.run_epoch(i, True)
        losses.append(float(loss_values["total"]))

        # this block can be edited to save/log in another way
        # this simply displays some outputs with matplotlib
        fig, (ax0, ax1) = plt.subplots(1, 2) 
        recon = outputs["depth"].detach().cpu().squeeze()
        vmin, vmax = recon.min(), recon.max()
        reconfig = ax1.imshow(recon, cmap='turbo', vmin = vmin, vmax=vmax)
        ax0.imshow(run_manager.reference_image.cpu().squeeze(), cmap='gray')
        
        cbar = plt.colorbar(reconfig, ax=ax1, orientation='horizontal')
        cbar.set_label('Depth')

        ax1.set_title(f"height map, iteration {i}")
        ax0.set_title(f"reference image, frame {frame_number}")
        plt.savefig("skull-video-recon.png")
