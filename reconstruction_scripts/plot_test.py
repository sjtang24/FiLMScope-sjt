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
from scipy.ndimage import gaussian_filter


parser = argparse.ArgumentParser(description="Plot reconstruction with parameters.")
parser.add_argument("--downsampling", type=int, help="Downsampling Factor (1, 2, 4, 8)", default = 4)
parser.add_argument("--arrangement", type=str, help = "Camera Arrangement ('2x2 grid', '4x4 grid', 'all_cameras', 'narrow_sparse', 'wide_sparse')", default = '4x4 grid')
parser.add_argument("--iters", type=int, help="Number of iterations after calibration (10, 20, 30)", default = 10)
parser.add_argument("--rmse", action='store_true')
parser.add_argument("--ssim", action='store_true')
args = parser.parse_args()

if args.downsampling not in [1, 2, 4, 8]:
    sys.exit("Downsampling factor must be any of [1, 2, 4, 8]!")
if args.arrangement not in ['2x2 grid', '4x4 grid', 'all_cameras', 'narrow_sparse', 'wide_sparse']:
    sys.exit("Camera Arrangement must be any of ['2x2 grid', '4x4 grid', 'all_cameras', 'narrow_sparse', 'wide_sparse']!")
if args.iters not in [10, 20, 30]:
    sys.exit("Number of iterations after calibration step must be any of [10, 20, 30]")
if args.rmse == args.ssim:
    sys.exit("Must choose a single metric (ssim or rmse).")

metric = 'ssim' if args.ssim else 'rmse'

print(f"Animating reconstructions for {args.arrangement}, x{args.downsampling}, {args.iters} iterations...")
with open("data/iter-gold-standards.pkl", 'rb') as gold_standards_info_file:
    gold_standards_info_data = pickle.load(gold_standards_info_file)
gold_standards_info = gold_standards_info_data[1]['all_cameras']

df = pd.read_csv("ssim_dataset.csv")
df_to_consider = df[
    (df['downsampling'] == args.downsampling) & \
    (df['arrangement'] == args.arrangement) & \
    (df['iterations'] == args.iters)
]

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem)

img_h, img_w = None, None 
# extract the gold standards from file
gold_standards, reconstructions = None, None  
gs_path = Path("data/goldstandards")
recon_path = Path(f"data/{args.iters}/{args.arrangement}/{args.downsampling}")
all_gs_files = sorted(gs_path.glob(f'*.npy'), key = numeric_sort_key)
all_recon_files = sorted(recon_path.glob(f'*.npy'), key = numeric_sort_key)
all_gs_files_n = len(all_gs_files)
assert all_gs_files_n == len(all_recon_files)
for frame_num, gldstd in enumerate(tqdm(all_gs_files)):
    gold_standards_allit = np.load(f"data/goldstandards/{frame_num}.npy", mmap_mode = 'r')
    reconstructions_allit = np.load(f"data/{args.iters}/{args.arrangement}/{args.downsampling}/{frame_num}.npy", mmap_mode = 'r')
    _, img_h, img_w = gold_standards_allit.shape
    if frame_num == 0:
        gold_standards = np.zeros((all_gs_files_n, img_h, img_w))
        reconstructions = np.zeros((all_gs_files_n, img_h, img_w))
    gold_standards[frame_num, :, :] = gold_standards_allit[-1, :, :]
    reconstructions[frame_num, :, :] = reconstructions_allit[-1, :, :]

"""x = np.arange(img_w)
y = np.arange(img_h)
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(
    -y, -x, gold_standards[0, :, :], cmap='Blues', 
    linewidth=0, edgecolor ='none', alpha = 0.75
)

def update_surface(frame):
    ax.clear()
    ax.set_zlim(np.min(Zs), np.max(Zs))    
    ax.set_title(f'{frame}/{frame_num}')
    surf = ax.plot_surface(
        -y, -x, gold_standards[frame], 
        cmap='Blues', linewidth=0, edgecolor='none', alpha = 0.85
    )
    return surf 

ani = FuncAnimation(fig, update_surface, frames=frame_num, interval=100, blit=False)
plt.show()"""

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
rec_img = ax2.imshow(reconstructions[0, :, :], cmap = 'turbo')
gs_img = ax1.imshow(gold_standards[0, :, :], cmap = 'turbo')
ref_img = ax0.imshow(gold_standards_info[0]['reference'], cmap = 'gray')

fig.suptitle(f"Iteration 0/{frame_num} ({args.arrangement}, x{args.downsampling} w/ {args.iters} iterations)")

ax3.set_xlim(0, 600)
ax3.set_ylim(df_to_consider[metric].min(), df_to_consider[metric].max())

def update(frame):
    rec = reconstructions[frame, :, :]
    ref = gold_standards_info[frame]['reference']
    gs = gold_standards[frame, :, :]

    ref_img.set_data(ref)
    gs_img.set_data(gs)
    rec_img.set_data(rec)

    ssim_line.set_data(
        df_to_consider['frame_number'][:(frame+1)],
        df_to_consider[metric][:(frame+1)]
    )

    fig.suptitle(f"Iteration {frame}/{frame_num} ({args.arrangement}, x{args.downsampling} w/ {args.iters} iterations)")
    return [ref_img, gs_img, rec_img, ssim_line]

ani = FuncAnimation(fig, update, frames=all_gs_files_n, interval=10, blit=False, repeat=False)
plt.show()