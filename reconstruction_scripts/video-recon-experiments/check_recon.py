import pickle 
import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter, median_filter

def normalize(img, ep = 1e-8):
    return (img - img.min()) / (img.max() - img.min() + ep)

sample_name = 'eye_video'
recon_info_path = 'data/'
recon_path = 'recon/'
config = '1_4x4-grid_4'
#filepath = recon_path + sample_name + '/goldstandard.npy'
filepath_gs = recon_path + sample_name + '/goldstandard.npy'
filepath_recon = recon_path + sample_name + f'/{config}.npy'
run_info_file = recon_info_path + sample_name + '/iter-1.pkl'

with open(run_info_file, 'rb') as gold_standards_info_file:
    gold_standards_info_data = pickle.load(gold_standards_info_file)
#print(gold_standards_info_data[1].keys())
gold_standards_info = gold_standards_info_data[1]['4x4 grid']

gs_frames = np.load(filepath_gs, mmap_mode = 'r')
#print(gs_frames[:, :, :])
#recon_frames = np.load(filepath2, mmap_mode = 'r')
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

recon_frames = np.load(filepath_recon, mmap_mode = 'r')

#print(gold_standards_info[0])
rec_img = ax2.imshow(normalize(recon_frames[0, :, :]), cmap = 'turbo', vmin=0, vmax=1)
gs_img = ax1.imshow(normalize(gs_frames[0, :, :]), cmap = 'turbo', vmin=0, vmax=1)
ref_img = ax0.imshow(gold_standards_info[0]['reference'], cmap = 'gray')

fig.suptitle(f"Iteration 0")

def update(frame):
    rec = normalize(recon_frames[frame, :, :])
    ref = gold_standards_info[frame]['reference']
    gs = normalize(gs_frames[frame, :, :])

    ref_img.set_data(ref)
    gs_img.set_data(gs)
    rec_img.set_data(rec)


    fig.suptitle(f"Iteration {frame}")
    return [gs_img, gs_img, rec_img]

ani = FuncAnimation(fig, update, frames=np.arange(0, nframes), interval=1, blit=False, repeat=True)
#plt.show()
ani.save(f'{config}.gif', writer='pillow', fps=5)