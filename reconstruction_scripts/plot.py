import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
from datetime import datetime 
import imageio
import os
import pickle 
import pandas as pd 

time = input("Enter the time:")
sample_name = "knuckle_video"
filename = f'timings/video-reconstruction_{sample_name}.npy'
times = np.load(filename)
startup_times = np.load(f"timings/setup_{sample_name}.npy")
#print(times)
downsample_factors = np.array([1, 2, 4, 8, 16, 32])

# total runtime duration
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f"Downsampling Speedup for Sample {sample_name}")
ax1.plot(downsample_factors, np.sum(times, axis = 1), linestyle='-')
ax1.set_xscale('log', base = 2)
ax1.set_xlabel('Downsampling Factors')
ax1.set_title('Total Runtime (in seconds)')

# mean runtime duration
ax2.errorbar(downsample_factors, np.mean(times, axis = 1), yerr=np.std(times, axis = 1))
ax2.set_xscale('log', base = 2)
ax2.set_xlabel('Downsampling Factors')
ax2.set_title('Runtime per Frame (in seconds)')
fig.savefig(f"plots/{time}/timing/{sample_name}/runtime.png")

plt.figure()
plt.plot(downsample_factors, startup_times)
plt.title("Startup Time, by Downsampling Factors")
plt.xlabel('Downsampling Factors')
plt.ylabel("Time (in seconds)")
plt.savefig(f"plots/{time}/timing/{sample_name}/setup.png")

directory = Path(f"plots/{time}/heightmap/{sample_name}")
heightmaps_filenames = {downsample : sorted(file for file in directory.glob(f"*ds{downsample}.png")) for downsample in downsample_factors}

# try:
#     os.makedirs(f"animations/{sample_name}")
# except FileExistsError:
#     pass 

# for factor in downsample_factors:
#     images = [imageio.imread(f) for f in heightmaps_filenames[factor]]
#     imageio.mimsave(f'animations/animation_{factor}.gif', images, duration=0.2, loop = 0)  # duration is per frame in seconds

with open("rmses.pkl", 'rb') as rmse_file:
    rmses = pickle.load(rmse_file)

with open("ssim.pkl", 'rb') as ssim_file:
    ssims = pickle.load(ssim_file)

rmses_df = pd.DataFrame.from_dict(rmses, orient = 'columns')
ssims_df = pd.DataFrame.from_dict(ssims, orient = 'columns')

fig = plt.figure()
for col in rmses_df.columns:
    plt.plot(rmses_df.index, rmses_df[col], label=f'{col}')

plt.title("RMSE between Gold Standard and 3D Reconstructions\n(Each line corresponds to a downsampling factor)")
plt.xlabel("Iteration")
plt.ylabel("RMSE")
plt.legend(title="Downsampling Factors")
plt.tight_layout()
fig.savefig(f"rmses.png")

fig = plt.figure()
for col in ssims_df.columns:
    plt.plot(ssims_df.index, ssims_df[col], label=f'{col}')

plt.title("SSIM between Gold Standard and 3D Reconstructions\n(Each line corresponds to a downsampling factor)")
plt.xlabel("Iteration")
plt.ylabel("SSIM")
plt.legend(title="Downsampling Factors")
plt.tight_layout()
fig.savefig(f"ssims.png")