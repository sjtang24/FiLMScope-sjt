import pickle 
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, median_filter

filepath = '/data2/steven/goldstandard.npy'
filepath2 = '/data2/steven/1_4x4-grid_1.npy'
run_info_file = 'data/iter-4.pkl'

with open("data/iter-gold-standards.pkl", 'rb') as gold_standards_info_file:
    gold_standards_info_data = pickle.load(gold_standards_info_file)
gold_standards_info = gold_standards_info_data[1]['all_cameras']

gs_frames = np.load(filepath, mmap_mode = 'r')
recon_frames = np.load(filepath2, mmap_mode = 'r')
nframes, _, _ = gs_frames.shape

fig = plt.figure(figsize = (15, 4))

width_ratios = [1, 1, 1]
gs = GridSpec(1, 3, width_ratios=width_ratios)
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])

ax0.axis('off')
ax1.axis('off')
ax2.axis('off')

ax2.set_title("Reconstructed")
ax1.set_title("Gold Standard")
ax0.set_title("Reference Image")

sigma = (5,5)

rec_img = ax2.imshow(recon_frames[0, :, :], cmap = 'turbo')
gs_img = ax1.imshow(gs_frames[0, :, :], cmap = 'turbo')
ref_img = ax0.imshow(gold_standards_info[0]['reference'], cmap = 'gray')

fig.suptitle(f"Iteration 0")

def update(frame):
    rec = recon_frames[frame, :, :]
    ref = gold_standards_info[frame]['reference']
    gs = gs_frames[frame, :, :]

    ref_img.set_data(ref)
    gs_img.set_data(gs)
    rec_img.set_data(rec)

    fig.suptitle(f"Iteration {frame}")
    return [ref_img, gs_img, rec_img]

ani = FuncAnimation(fig, update, frames=np.arange(0, nframes), interval=25, blit=False, repeat=True)
plt.show()
