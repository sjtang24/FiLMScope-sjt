import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
from datetime import datetime 
import imageio
import os
import pickle 
import pandas as pd 

sample_name = "knuckle_video"
time = "2025-05-25_12:52:06"
directory = Path(f"plots/{time}/heightmap/{sample_name}")

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem)

downsample_factors = np.array([1, 2, 4, 8, 16, 32])
camera_arrangements = ['all_cameras', 'wide_sparse', 'narrow_sparse', '2x2 grid', '4x4 grid']
heightmaps_filenames = {
    downsample : {
        camera_type : sorted((file for file in directory.glob(f"*ds{downsample}_{camera_type}*.png")), key = numeric_sort_key)
        for camera_type in camera_arrangements
    }
    for downsample in downsample_factors
}

pause_frames = 20
frame_duration = 0.1  # seconds (100 ms per frame)

for factor in downsample_factors:
    for arrangement in camera_arrangements:
        images = [imageio.imread(f) for f in heightmaps_filenames[factor][arrangement]]
        imageio.mimsave(
            f'animations/animation_{factor}_{arrangement}.gif',
            images,
            duration=frame_duration,
            loop=0
        )