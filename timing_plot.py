import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('timing.csv')
df['downsampling'] = df['binning'] * df['subsampling']
df['config'] = (
    "x" + df['binning'].astype(str) + " bin, " +
    "x" + df['subsampling'].astype(str) + " sub\n(x" +
      df['downsampling'].astype(str) + ' d.s.)'
)
print(df)
summary_stats = df.groupby(['downsampling', 'binning', 'subsampling' ,'config']).mean().reset_index()
print(summary_stats[['downsampling', 'binning', 'subsampling', 'acq', 'vol', 'recon', 'latency', 'throughput']])
summary_stats = df.groupby(['downsampling', 'binning', 'subsampling' ,'config']).std().reset_index()
print(summary_stats[['downsampling', 'binning', 'subsampling', 'acq', 'vol', 'recon', 'latency', 'throughput']])
summary_stats['config'] = (
    "x" + summary_stats['binning'].astype(str) + " bin, " +
    "x" + summary_stats['subsampling'].astype(str) + " sub\n(x" +
      summary_stats['downsampling'].astype(str) + ' d.s.)'
)

summary_stats[['config', 'acq', 'vol', 'recon']].plot(kind='bar', stacked=True, figsize=(10, 6))
plt.xticks(ticks=range(len(summary_stats.index)), labels=summary_stats['config'])
plt.title('Average Reconstruction Latency for x4 and x8 binning and subsampling')
plt.xlabel('Downsampling Configuration')
plt.ylabel('Reconstruction Latency by Component (seconds)')
plt.legend(title='Algorithm Component')
plt.xticks(rotation=0)
plt.show()

# Create boxplot: x is the discrete category, y is the measurement
sns.boxplot(data=df, x='config', y='throughput')
plt.xticks(rotation=0)
plt.title('Throughput for x4 and x8 binning and subsampling')
plt.xlabel('Downsampling Configuration')
plt.ylabel('Throughput (Hz)')
plt.tight_layout()
plt.show()

sns.boxplot(data=df, x='config', y='latency')
plt.xticks(rotation=0)
plt.title('Latency for x4 and x8 binning and subsampling')
plt.xlabel('Downsampling Configuration')
plt.ylabel('Latency (seconds)')
plt.tight_layout()
plt.show()