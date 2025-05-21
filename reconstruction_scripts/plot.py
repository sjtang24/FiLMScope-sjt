import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
import imageio
import os

sample_name = "knuckle_video"
filename = f'timings/video-reconstruction_{sample_name}.npy'
times = np.load(filename)
#print(times)
downsample_factors = np.array([2, 4, 7, 8, 16, 32])

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
fig.savefig(f"plots/timing/{sample_name}/runtime.png")

directory = Path(f"plots/heightmap/{sample_name}")
heightmaps_filenames = {downsample : sorted(file for file in directory.glob(f"*ds{downsample}.png")) for downsample in downsample_factors}

try:
    os.makedirs(f"animations/{sample_name}")
except FileExistsError:
    pass 

for factor in downsample_factors:
    images = [imageio.imread(f) for f in heightmaps_filenames[factor]]
    imageio.mimsave(f'animations/animation_{factor}.gif', images, duration=0.2, loop = 0)  # duration is per frame in seconds


