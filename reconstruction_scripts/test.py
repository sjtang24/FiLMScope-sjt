import pickle
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import numpy as np
from pathlib import Path 
from skimage.metrics import structural_similarity
from tqdm import tqdm
import pandas as pd 
import argparse
import sys

parser = argparse.ArgumentParser(description="Plot reconstruction with parameters.")
parser.add_argument("--calc", action='store_true', help="Recalculate Metric")
parser.add_argument("--ssim", action='store_true', help="Recalculate Metric")
parser.add_argument("--rmse", action='store_true', help="Recalculate Metric")
args = parser.parse_args()

if args.ssim == args.rmse:
    sys.exit("Choose one of SSIM or RMSE!")
metric_choice = 'ssim' if args.ssim else 'rmse'

def numeric_sort_key(path):
    # Extract all numbers in the filename and return them as a tuple of ints
    return int(path.stem)

downsampling_factors = [1, 2, 4, 8]
arrangements = ['all_cameras', '4x4 grid', 'wide_sparse', 'narrow_sparse', '2x2 grid']
iters = [10, 20, 30]

if args.calc:
    # extract the gold standards from file
    gold_standards = None 
    gs_path = Path("data/goldstandards")
    all_gs_files = sorted(gs_path.glob(f'*.npy'), key = numeric_sort_key)
    all_gs_files_n = len(all_gs_files)
    for frame_num, gldstd in enumerate(all_gs_files):
        gold_standards_allit = np.load(f"data/goldstandards/{frame_num}.npy", mmap_mode = 'r')
        _, img_h, img_w = gold_standards_allit.shape
        if frame_num == 0:
            gold_standards = np.zeros((all_gs_files_n, img_h, img_w))
        gold_standards[frame_num, :, :] = gold_standards_allit[-1, :, :]

    ssim_records = []

    for arrangement in arrangements:
        for downsampling in downsampling_factors:
            for iterations in iters:
                print(f"Gathering data for {arrangement}, x{downsampling} (iterations = {iterations})")
                files_dir = Path(f"data/{iterations}/{arrangement}/{downsampling}")
                all_files = sorted(files_dir.glob(f"*.npy"), key=numeric_sort_key)
                #print(all_files)
                for frame_num, npy_files in enumerate(tqdm(all_files)):
                    reconstruction_ev_allit = np.load(npy_files, mmap_mode = 'r') 
                    ssim_records.append(
                        {
                            'frame_number' : frame_num,
                            'arrangement' : arrangement,
                            'downsampling' : downsampling,
                            'iterations' : iterations,
                            'ssim' : structural_similarity(
                                gold_standards[frame_num, :, :], 
                                reconstruction_ev_allit[-1, :, :], 
                                full = False
                            ),
                            'rmse' : np.sqrt(
                                np.mean(
                                    np.square(
                                        gold_standards[frame_num, :, :] - reconstruction_ev_allit[-1, :, :]
                                    )
                                )
                            )
                        }
                    )

    ssim_dataset = pd.DataFrame(ssim_records)
else:
    ssim_dataset = pd.read_csv("ssim_dataset.csv")

def plot_data(metric):
    n_rows = len(downsampling_factors)
    n_cols = len(arrangements)

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(4.2 * n_cols, 3.2 * n_rows),
        sharex=True,
        sharey=metric == 'ssim',
        constrained_layout=False
    )

    axes = axes.flatten()
    plot_idx = 0

    for downsampling in downsampling_factors:
        for arrangement in arrangements:
            ax = axes[plot_idx]
            df = ssim_dataset[
                (ssim_dataset["arrangement"] == arrangement) & 
                (ssim_dataset["downsampling"] == downsampling)
            ]

            if not df.empty:
                for stop_after, stop_after_df in df.groupby("iterations"):
                    ax.plot(
                        stop_after_df['frame_number'],
                        stop_after_df[metric],
                        linewidth = 0.5,
                        label=f"{stop_after}"
                    )
                ax.set_title(f"{arrangement}\n×{downsampling}", fontsize=10)
                if plot_idx % n_cols == 0:
                    ax.set_ylabel(metric.upper(), fontsize=9)
                if plot_idx >= n_cols * (n_rows - 1):
                    ax.set_xlabel("Frame", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, linewidth=0.5, linestyle='--', alpha=0.6)
            else:
                ax.set_visible(False)

            plot_idx += 1

    # Shared legend above all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Stop After",
        loc='upper left',
        #bbox_to_anchor=(0.5, 1.03),
        ncol=len(iters),
        fontsize="small",
        title_fontsize="small"
    )

    fig.set_constrained_layout(False)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    fig.suptitle(f"{metric.upper()} Change Over Frames", fontsize=16, y=0.98)
    plt.show()


plot_data(metric_choice)

if args.calc:
    ssim_dataset.to_csv('ssim_dataset.csv', index=False)

"""print("Opening the file....")
with open(f"data/iter-10.pkl", "rb") as itfile:
iteration_data = pickle.load(itfile)
print("Finished opening the file and extracting data...")"""
"""
img_h, img_w = recon.shape
X, Y = np.meshgrid(np.arange(img_w), np.arange(img_h))

fig = go.Figure(data=[go.Surface(z=recon, x=-Y, y=-X, colorscale='Blues')])
fig.update_layout(
width = 900, height = 900,
scene_camera=dict(eye=dict(x=0, y=0, z=2)),
scene=dict(
xaxis=dict(visible=False),
yaxis=dict(visible=False),
zaxis=dict(visible=False)
)    
)
fig.write_image(f"data/3d-{frame_num}.png")"""
#print(f"Frame {frame_num} | SSIM = {ssim}")
"""plt.plot(np.arange(len(all_files)), ssims)
plt.savefig("data/plot.png")"""

"""heightmaps = gstds["all_cameras"][2][0]['iterations']
reference = gstds["all_cameras"][2][0]['reference']
expanded_heightmap = heightmaps[-1, :, :]

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
ax1.imshow(reference, cmap='gray')
ax0.imshow(reference, cmap='gray')

ax2.set_title(f"Reconstructed")
ax1.set_title(f"Gold Standard")
ax0.set_title(f"Reference Image")
#plt.savefig(f"data/0.png")
plt.show()

"""