# This code visualizes the 3d reconstruction, either as a surface or as a heightmap, depending on 
# parameters.

import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D 
import numpy as np
import pandas as pd
from pathlib import Path 
from skimage.metrics import structural_similarity
from tqdm import tqdm
import pandas as pd 
import argparse
import sys
from scipy.ndimage import gaussian_filter, median_filter

parser = argparse.ArgumentParser(description="Plot reconstruction with parameters.")
parser.add_argument("--downsampling", type=int, help="Downsampling Factor (1, 2, 4, 8)", default = 1)
parser.add_argument("--arrangement", type=str, help = "Camera Arrangement ('2x2-grid', '4x4-grid', 'all_cameras', 'narrow_sparse', 'wide_sparse')", default = '4x4-grid')
parser.add_argument("--iters", type=int, help="Number of iterations after calibration (10, 20, 30)", default = 1)
parser.add_argument("--rmse", action='store_true')
parser.add_argument("--ssim", action='store_true')
parser.add_argument("--surface", action='store_true')
parser.add_argument("--heightmap", action='store_true')
args = parser.parse_args()

if args.downsampling not in [1, 2, 4, 8]:
    sys.exit("Downsampling factor must be any of [1, 2, 4, 8]!")
if args.arrangement not in ['2x2-grid', '4x4-grid', 'all_cameras', 'narrow_sparse', 'wide_sparse']:
    sys.exit("Camera Arrangement must be any of ['2x2-grid', '4x4-grid', 'all_cameras', 'narrow_sparse', 'wide_sparse']!")
if args.iters not in [1, 2, 4, 8, 10, 20, 30]:
    sys.exit("Number of iterations after calibration step must be any of [10, 20, 30]")
if args.rmse == args.ssim:
    sys.exit("Must choose a single metric (ssim or rmse).")
if args.surface == args.heightmap:
    sys.exit("Must choose a single representation ('surface' or 'heightmap')")

metric = 'ssim' if args.ssim else 'rmse'

print(f"Animating reconstructions for {args.arrangement}, x{args.downsampling}, {args.iters} iterations...")
with open("data/iter-gold-standards.pkl", 'rb') as gold_standards_info_file:
    gold_standards_info_data = pickle.load(gold_standards_info_file)
gold_standards_info = gold_standards_info_data[1]['all_cameras']

df = pd.read_csv("metrics_dataset.csv")
df_to_consider = df[
    (df['downsampling'] == args.downsampling) & \
    (df['arrangement'] == ' '.join(args.arrangement.split('-'))) & \
    (df['iterations'] == args.iters)
]

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem)

gold_standards = np.load(f"/data2/steven/goldstandard.npy", mmap_mode = 'r')
reconstructions = np.load(f"/data2/steven/{args.iters}_{args.arrangement}_{args.downsampling}.npy", mmap_mode = 'r')
frame_num, img_w, img_h = reconstructions.shape 
size = (5, 5)

if args.surface:
    x = np.arange(img_w)
    y = np.arange(img_h)
    x, y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(
        -y, -x, reconstructions[0, :, :], cmap='Blues', 
        linewidth=0, edgecolor ='none', alpha = 0.75
    )
        
    def update_surface(frame):
        ax.clear()
        ax.set_zlim(np.min(reconstructions[frame, :, :]), \
                    np.max(reconstructions[frame, :, :]))    
        ax.set_title(f'{frame}/{frame_num}')
        surf = ax.plot_surface(
            -y, -x, median_filter(reconstructions[frame, :, :], size), 
            cmap='Blues', linewidth=0, edgecolor='none', alpha = 0.85
        )
        return surf 

    ani = FuncAnimation(fig, update_surface, frames=frame_num, interval=100, blit=False)
    plt.show()
else: 
    fig = plt.figure(figsize = (15, 4))

    width_ratios = [1, 1, 1, 1.5]
    gs = GridSpec(1, 4, width_ratios=width_ratios)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax3 = fig.add_subplot(gs[3])

    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')

    ax3.set_title(f"{metric} across Frames")
    ax3.set_xlabel("Frame Number")

    ax2.set_title("Reconstructed")
    ax1.set_title("Gold Standard")
    ax0.set_title("Reference Image")

    ssim_line, = ax3.plot([], [], color='blue')  # empty line initially
    """rec_img = ax2.imshow(median_filter(reconstructions[0, :, :], size), cmap = 'turbo')
    gs_img = ax1.imshow(median_filter(gold_standards[0, :, :], size), cmap = 'turbo')
    ref_img = ax0.imshow(median_filter(gold_standards_info[0]['reference'], size), cmap = 'gray')"""
    rec_img = ax2.imshow(median_filter(reconstructions[0, :, :], size), cmap = 'turbo')
    gs_img = ax1.imshow(median_filter(gold_standards[0, :, :], size), cmap = 'turbo')

    ref = gold_standards_info[0]['reference']
    min_gs = gold_standards_info[0]['reference'].min()
    max_gs = gold_standards_info[0]['reference'].max()
    scaled = (ref - min_gs) / (max_gs - min_gs)
    ref_img = ax0.imshow(scaled, cmap = 'gray')

    fig.suptitle(f"Iteration 0/{frame_num} ({args.arrangement}, x{args.downsampling} w/ {args.iters} iterations)")

    ax3.set_xlim(0, 600)
    ax3.set_ylim(df_to_consider[metric].min(), df_to_consider[metric].max())

    def update(frame):
        rec = reconstructions[frame, :, :]
        ref = gold_standards_info[frame]['reference']
        gs = gold_standards[frame, :, :]

        ref = (ref - ref.min()) / (ref.max() - ref.min())
        ref_img.set_data(ref)
        gs_img.set_data(gs)
        rec_img.set_data(rec)

        ssim_line.set_data(
            df_to_consider['frame_number'][:(frame+1)],
            df_to_consider[metric][:(frame+1)]
        )

        fig.suptitle(f"Iteration {frame}/{frame_num} ({args.arrangement}, x{args.downsampling} w/ {args.iters} iterations)")
        return [ref_img, gs_img, rec_img, ssim_line]

    ani = FuncAnimation(fig, update, frames=range(frame_num + 1), interval=10, blit=False, repeat=True)
    plt.show()
