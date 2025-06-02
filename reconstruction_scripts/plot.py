import numpy as np
from pathlib import Path 
from matplotlib import pyplot as plt
from datetime import datetime 
import imageio
import os
import pickle 
import pandas as pd 

time = input("Enter the time:")
if len(time) == 0:
    time = '2025-05-28_16:13:35'
sample_name = "knuckle_video"
filename = f'timings/video-reconstruction_{sample_name}.npy'
times = np.load(filename, allow_pickle = True).item()
startup_times = np.load(f"timings/setup_{sample_name}.npy", allow_pickle = True).item()
downsample_factors = np.array([1, 2, 4, 8, 16, 32])

# total runtime duration
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle(f"Downsampling Speedup for Sample {sample_name}")
for arrangement in times:
    ax1.plot(downsample_factors, np.sum(times[arrangement], axis = 1), linestyle='-', label = arrangement)
    ax2.errorbar(downsample_factors, np.mean(times[arrangement], axis = 1), yerr=np.std(times[arrangement], axis = 1), label = arrangement)
ax1.set_xscale('log', base = 2)
ax1.set_xlabel('Downsampling Factors')
ax1.set_title('Total Runtime (in seconds)')
ax2.set_xscale('log', base = 2)
ax2.set_xlabel('Downsampling Factors')
ax2.set_title('Runtime per Frame (in seconds)')
ax2.legend()
fig.savefig(f"plots/{time}/timing/{sample_name}/runtime.png")

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)  # Create an Axes object

for arrangement in times:
    ax.plot(downsample_factors, startup_times[arrangement], label=arrangement)

ax.set_xscale('log', base=2)
ax.set_title("Startup Time, by Downsampling Factors")
ax.set_xlabel('Downsampling Factors')
ax.set_ylabel("Time (in seconds)")
ax.legend()
fig.savefig(f"plots/{time}/timing/{sample_name}/setup.png")

with open("metrics.pkl", 'rb') as metrics_file:
    metrics = pickle.load(metrics_file)
metrics = metrics[420]
print(metrics)


metrics_dict = {
    arrangement : pd.DataFrame.from_dict(metrics[arrangement], orient = 'columns') for arrangement in metrics
}
 
num_camera_types = len(times.keys())

fig_rmse, axs_rmse = plt.subplots(1, num_camera_types, sharey = True, figsize=(30, 6))
fig_ssim, axs_ssim = plt.subplots(1, num_camera_types, sharey = True, figsize=(30, 6))

for arrangements_i, arrangement in enumerate(times):
    metrics_df = metrics_dict[arrangement]

    for col in metrics_df.columns:
        axs_rmse[arrangements_i].plot(metrics_df[col]['duration'], metrics_df[col]['RMSE'], label=f'{col}')
        axs_ssim[arrangements_i].plot(metrics_df[col]['duration'], metrics_df[col]['SSIM'], label=f'{col}')

    if arrangements_i == 0:
        axs_rmse[arrangements_i].set_ylabel("RMSE")
        axs_ssim[arrangements_i].set_ylabel("SSIM")
    elif arrangements_i == num_camera_types // 2:
        axs_rmse[arrangements_i].set_xlabel("Time (in seconds)")  
        axs_ssim[arrangements_i].set_xlabel("Time (in seconds)")

    axs_rmse[arrangements_i].set_title(f"{arrangement}")
    axs_ssim[arrangements_i].set_title(f"{arrangement}")

    axs_rmse[arrangements_i].legend()
    axs_ssim[arrangements_i].legend()
    axs_rmse[arrangements_i].set_xlim(0, 25)
    axs_ssim[arrangements_i].set_xlim(0, 25)

fig_rmse.suptitle("RMSE Between Gold Standard and Height Reconstructions, Across Downsampling Factors")
fig_rmse.tight_layout()
fig_rmse.savefig(f"plots/{time}/rmses-og.png")

fig_ssim.suptitle("Structural Similarity Between Gold Standard and Height Reconstructions, Across Downsampling Factors")
fig_ssim.tight_layout()
fig_ssim.savefig(f"plots/{time}/ssims-og.png")

for arrangement in metrics.keys():
    metric_by_arrangements = metrics[arrangement]
    for downsampling in metric_by_arrangements.keys():
        metric_by_arrds = metric_by_arrangements[downsampling]
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(metric_by_arrds['bestimage_RMSE'], cmap = 'turbo')
        axs[0].axis('off')
        axs[1].imshow(metric_by_arrds['bestimage_SSIM'], cmap = 'turbo')
        axs[1].axis('off')
        axs[0].set_title(f"RMSE = {round(metric_by_arrds['best_RMSE'], 3)}\nIteration {metric_by_arrds['best_RMSE_it']} (after {round(metric_by_arrds['best_RMSE_time'], 2)}s)")
        axs[1].set_title(f"SSIM = {round(metric_by_arrds['best_SSIM'], 3)}\nIteration {metric_by_arrds['best_SSIM_it']} (after {round(metric_by_arrds['best_SSIM_time'], 2)}s)")
        fig.suptitle(f"Best Images\nDownsampling Factor: {downsampling}\nCamera Arrangement: {arrangement}")
        fig.savefig(f"plots/{time}/best_{arrangement}_{downsampling}-og.png")