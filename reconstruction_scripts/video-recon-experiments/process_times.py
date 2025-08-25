import pickle 
from pathlib import Path
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import os
import re

def numeric_sort_key(path):
    suffix = path.stem.split('-')[-1]
    if str.isnumeric(suffix):
        return int(suffix)
    return 0

if not os.path.exists('durations.csv'):
    print("'durations.csv' does not exists")

    iter_path = Path("data")
    iteration_files = sorted(iter_path.glob(f'iter-*.pkl'), key = numeric_sort_key)

    records = []
    # Downsampling > Arrangement > Frame Number > Duration
    for key, it_file in enumerate(iteration_files):
        print(f"Processing {it_file.name}")
        match = re.match(r"iter-(.+)\.pkl", it_file.name).group(1)
        with open(it_file, 'rb') as iterfile:
            durations = pickle.load(iterfile)
            for ds in durations.keys():
                for arr in durations[ds].keys():
                    for frame in durations[ds][arr].keys():
                        if frame == 'setup_time':
                            records.append(
                                {
                                    'stop_after' : match,
                                    'downsampling' : ds,
                                    'arrangement' : arr,
                                    'duration' : durations[ds][arr]['setup_time']
                                }
                            ) 
                        else:
                            records.append(
                                {
                                    'stop_after' : match,
                                    'downsampling' : ds,
                                    'arrangement' : arr,
                                    'frame_number' : frame,
                                    'duration' : durations[ds][arr][frame]['duration'][-1]
                                }
                            )
                    print(f"Finished Downsampling {ds} | Arrangement {arr}")

    duration_dataset = pd.DataFrame(records)
    duration_dataset.to_csv('durations.csv', index=False)
else:
    metric = "rmse"
    def plot_with_custom_labels(plot, title, filename):
        plot.fig.subplots_adjust(top=0.9)
        plot.fig.suptitle(
            title,
            fontsize=16,
            fontweight='bold'
        )
        plot.set_titles("{col_name}, x{row_name} Downsampling")
        plot.savefig(filename)

    AFTER = 200 
    # Load and merge datasets
    duration_dataset = pd.read_csv('durations.csv')
    metrics_dataset = pd.read_csv('metrics_dataset.csv')

    setup_times_df = duration_dataset[duration_dataset['frame_number'].isna()]

    metrics_dataset_renamed = metrics_dataset.rename(columns={'iterations': 'stop_after'})
    factors = ['downsampling', 'arrangement', 'frame_number', 'stop_after']
    metrics_dataset_renamed['stop_after'] = metrics_dataset_renamed['stop_after'].astype(str)

    combined = duration_dataset.merge(metrics_dataset_renamed, how='left', on=factors)
    combined.fillna({'ssim': 1, 'rmse': 0, 'filter_time': 0}, inplace=True)
    combined['total_duration'] = combined['duration'] + combined['filter_time']
    combined = combined[combined['frame_number'] >= AFTER]
    combined_metrics = combined.groupby(['downsampling', 'arrangement', 'stop_after'])[[metric, 'duration']].mean().reset_index()
    combined_metrics.to_csv('combined.csv', index=False)

    # Appearance maps
    stopafter_cmap = {
        'gold-standards': 'black',
        '10': 'red',
        #'20': 'blue',
        #'30': 'green',
        '1': 'blue',
        #'2': 'purple',
        '4': 'green',
        #'8': 'magenta'
    }

    arrangement_mmap = {
        'all_cameras': '*',
        '4x4 grid': 's',
        '2x2 grid': 'v',
        'wide_sparse': 'o',
        'narrow_sparse': '.'
    }

    ds_alphamap = {
        '1': 1,
        '2': 0.5,
        '4': 0.25,
        '8': 0.125
    }

    # Legends
    downsampling_legend = [
        mlines.Line2D([], [], color=color, marker='o', linestyle='None', label=f'{stop}', markeredgewidth=0)
        for stop, color in stopafter_cmap.items()
    ]

    arrangement_legend = [
        mlines.Line2D([], [], color='black', marker=marker, linestyle='None', label=arr, markeredgewidth=0)
        for arr, marker in arrangement_mmap.items()
    ]

    stopafter_legend = [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', label=f'x{ds}', alpha=alpha, markeredgewidth=0)
        for ds, alpha in ds_alphamap.items()
    ]

    # Plot
    plt.figure(figsize=(8, 6))

    for (ds, arr, stop), group in combined_metrics.groupby(['downsampling', 'arrangement', 'stop_after']):
        if stop not in ['2', '8', '20', '30']:
            plt.scatter(
                group['duration'],
                group[metric],
                label=f'{arr}, x{ds} ({stop})',
                marker=arrangement_mmap.get(arr, 'o'),
                color=stopafter_cmap.get(stop, 'grey'),
                alpha=ds_alphamap.get(str(ds), 0.5),
                edgecolors='none'
            )
    plt.xscale('log')
    plt.xlabel('Duration (in log seconds)')
    plt.ylabel(metric)
    plt.title(f'Average Per-Frame Duration vs {metric} for Various Configurations')

    # Add separate legends
    legend1 = plt.legend(handles=downsampling_legend, title='Stop After', loc='upper right', bbox_to_anchor=(1, 1))
    plt.gca().add_artist(legend1)

    legend2 = plt.legend(handles=arrangement_legend, title='Arrangement', loc='upper right', bbox_to_anchor=(0.8, 1))
    plt.gca().add_artist(legend2)

    legend3 = plt.legend(handles=stopafter_legend, title='Downsampling', loc='upper right', bbox_to_anchor=(0.55, 1))

    plt.tight_layout()
    plt.savefig(f"plots/avg-duration-rmse-{AFTER}-server.png")

    # Boxplots
    for filt in [None, 0]:
        duration_dataset_filt = duration_dataset
        if filt is not None:
            duration_dataset_filt = duration_dataset[duration_dataset['frame_number'] != filt]
        
        boxplots = sb.catplot(
            x="stop_after", y="duration", row="downsampling", col="arrangement",
            data=duration_dataset_filt[duration_dataset['frame_number'].notna()], kind="box", sharey=False, sharex=False, 
            height=2, aspect=1.5
        )
        
        incl = 'including' if filt is None else 'excluding'
        plot_with_custom_labels(
            boxplots, 
            f'Durations (sec) for Various Configurations ({incl} warmup steps)',
            f"plots/{incl}-durations-og.png"
        )

        barplot = sb.catplot(
            x="stop_after", y="duration", row="downsampling", col="arrangement",
            data=setup_times_df, kind="bar", sharey=False, sharex=False, 
            height=2, aspect=1.5
        )
        
        plot_with_custom_labels(barplot, f'Setup Times for Various Configurations', 
            "plots/setup-times-og.png"
        )

