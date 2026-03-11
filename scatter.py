import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfs = []
for seq in ['mh_01', 'mh_03', 'mh_05']:
    df = pd.read_csv(f'results/{seq}_uncertainty250.csv')
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
mean_e = df['error_m'].mean()
std_e = df['error_m'].std()
df = df[df['error_m'] < mean_e + 3 * std_e].copy()

features = [
    "mean_inlier_track_length", "condition_estimate", "match_std_x",
    "depth_mean", "depth_std", "img_contrast", "depth_range",
    "img_brightness", "match_spread_normalized", "mean_inverse_depth",
    "median_inlier_reproj_error", "img_blur_score", "mean_inlier_ba_error"
]

fig, axes = plt.subplots(4, 4, figsize=(16, 14))
axes = axes.flatten()

for i, feat in enumerate(features):
    ax = axes[i]
    ax.scatter(df[feat], df['error_m']*100, alpha=0.05, s=5, color='steelblue')
    ax.set_xlabel(feat, fontsize=7)
    ax.set_ylabel('error (cm)', fontsize=7)
    ax.set_title(feat, fontsize=8)

for j in range(len(features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('results/feature_scatter.png', dpi=120, bbox_inches='tight')
print("Saved to results/feature_scatter.png")