import pickle 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

earlystop = [10, 20, 30]
records = []

arrangements = None 
downsampling_factors = None 

for stoppoint_i, stoppoint in enumerate(earlystop):
    with open(f"data/metrics-{stoppoint}.pkl", "rb") as metrics_file:
        metrics = pickle.load(metrics_file)
    
    if stoppoint_i == 0:
        arrangements = metrics.keys()

    for arrangement_i, arrangement in enumerate(arrangements):
        if arrangement_i == 0:
            downsampling_factors = metrics[arrangement].keys()
        for downsampling in downsampling_factors:
            if arrangement == "all_cameras" and downsampling == 1:
                continue 
            for frame_no in metrics[arrangement][downsampling].keys():
                if frame_no == 'setup_time':
                    continue 
                records.append(
                    {
                        "stop_after" : stoppoint,
                        "arrangement" : arrangement,
                        "downsampling" : downsampling,
                        "frame_no" : frame_no,
                        "SSIM" : metrics[arrangement][downsampling][frame_no]["SSIM"][-1],
                        "RMSE" : metrics[arrangement][downsampling][frame_no]["RMSE"][-1]
                    }
                )
ssim_dataset = pd.DataFrame(records)

fig, axes = plt.subplots(nrows=len(downsampling_factors), 
                         ncols=len(arrangements), 
                         figsize=(5 * len(arrangements), 4 * len(downsampling_factors)),
                         sharex=True, sharey=True)

axes = axes.flatten()
plot_idx = 0

for downsampling in downsampling_factors:
    for arrangement in arrangements:
        ax = axes[plot_idx]
        df = ssim_dataset[
            (ssim_dataset["arrangement"] == arrangement) & 
            (ssim_dataset["downsampling"] == downsampling)
        ]
        #print(f"{arrangement} | {downsampling}")
        #print(ssim_dataset)
        if not df.empty:
            for stop_after, stop_after_df in df.groupby("stop_after"):
                ax.plot(stop_after_df["frame_no"], stop_after_df["SSIM"], label=stop_after)
            ax.set_title(f"{arrangement}, x{downsampling} downsampling")
            ax.set_xlabel("Frame Number")
            ax.set_ylabel("SSIM")
            ax.grid(True)
            ax.legend(title="Stop After", fontsize="small")
        else:
            ax.set_visible(False)
        
        plot_idx += 1

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # leave space for suptitle
plt.suptitle("SSIM Change over Frames", fontsize=16)
plt.savefig("det.png")
plt.show()


# make some videos (including 3d with plotly)