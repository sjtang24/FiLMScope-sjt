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
path_dir = "plots/3dheightmap"
directory = Path(path_dir)

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem.split("-")[-1])
"""
downsample_factors = np.array([1, 2, 4, 8, 16, 32])
camera_arrangements = ['all_cameras', 'wide_sparse', 'narrow_sparse', '2x2 grid', '4x4 grid']
heightmaps_filenames = {
    downsample : {
        camera_type : sorted((file for file in directory.glob(f"*ds{downsample}_{camera_type}*.png")), key = numeric_sort_key)
        for camera_type in camera_arrangements
    }
    for downsample in downsample_factors
}
"""

frames = sorted(
    (file for file in directory.glob(f"3d-*.png")), 
    key = numeric_sort_key
)
pause_frames = 20
frame_duration = 0.5  # seconds (100 ms per frame)

"""for factor in downsample_factors:
    for arrangement in camera_arrangements:"""

images = [imageio.imread(f) for f in frames]

imageio.mimsave(
    f'animations/animation.gif',
    images,
    duration=frame_duration,
    loop=0
)