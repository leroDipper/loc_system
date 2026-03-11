"""
Feature importance analysis for Bayesian localization error prediction.
Uses Random Forest + correlation matrix on 250/350/550 keypoint versions only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# LOAD DATA (250, 350, 550 only — 100 excluded as not representative of deployment)
# ============================================================================

print("Loading data...")

dfs = []
for seq in ['mh_01', 'mh_03', 'mh_05']:
    for kp in [250, 350, 550]:
        df = pd.read_csv(f'results/{seq}_uncertainty{kp}.csv')
        df['source'] = f'{seq}_{kp}'
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
print(f"Total frames: {len(df)}")

# ============================================================================
# ALL CANDIDATE FEATURES (everything available, commented or not in training)
# ============================================================================

candidate_features = [
    "n_matches",
    "n_inliers",
    "depth_mean",
    "depth_std",
    "depth_relative_std",
    "depth_range",
    "match_spread_normalized",
    "match_std_x",
    "match_std_y",
    "n_quadrants_active",
    "quadrant_entropy",
    "mean_inverse_depth",
    "condition_estimate",
    "mean_inlier_track_length",
    "median_inlier_track_length",
    "mean_inlier_ba_error",
    "median_inlier_ba_error",
    "frac_high_quality_inliers",
    "img_blur_score",
    "img_brightness",
    "img_contrast",
    "img_edge_density",
    "img_histogram_uniformity",
    "mean_inlier_reproj_error",
    "median_inlier_reproj_error", 
    "std_inlier_reproj_error",
]

# Only keep features that exist in the data
available_features = [f for f in candidate_features if f in df.columns]
missing = [f for f in candidate_features if f not in df.columns]
if missing:
    print(f"Missing from data (skipped): {missing}")

print(f"Analysing {len(available_features)} features")

# ============================================================================
# PREPARE DATA
# ============================================================================

df_clean = df[available_features + ['error_m']].dropna()

# 3-sigma outlier removal (same as training)
mean_e = df_clean['error_m'].mean()
std_e = df_clean['error_m'].std()
df_clean = df_clean[df_clean['error_m'] < mean_e + 3 * std_e]
print(f"After outlier removal: {len(df_clean)} frames")

X = df_clean[available_features].values
y = df_clean['error_m'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ============================================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================

print("\nFitting Random Forest...")
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

r2 = rf.score(X_test, y_test)
print(f"Random Forest R² on test set: {r2:.3f}")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature Importance Ranking:")
print(f"{'Rank':<6} {'Feature':<35} {'Importance':<10}")
print("-" * 55)
for rank, idx in enumerate(indices):
    print(f"{rank+1:<6} {available_features[idx]:<35} {importances[idx]:.4f}")

# ============================================================================
# CORRELATION MATRIX (features vs error + inter-feature)
# ============================================================================

df_corr = df_clean[available_features + ['error_m']].copy()
corr_with_error = df_corr.corr()['error_m'].drop('error_m').sort_values(key=abs, ascending=False)

print("\nCorrelation with error_m (absolute, ranked):")
print(f"{'Feature':<35} {'Correlation':<10}")
print("-" * 45)
for feat, val in corr_with_error.items():
    print(f"{feat:<35} {val:+.4f}")

# ============================================================================
# PLOTS
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: Feature importances
ax1 = axes[0]
sorted_importances = importances[indices]
sorted_features = [available_features[i] for i in indices]
colors = ['#2ecc71' if f in [
    "n_matches", "depth_range", "match_std_x",
    "mean_inlier_track_length", "n_inliers",
    "img_blur_score", "img_histogram_uniformity", "mean_inlier_ba_error"
] else '#3498db' for f in sorted_features]

bars = ax1.barh(range(len(sorted_features)), sorted_importances[::-1], color=colors[::-1])
ax1.set_yticks(range(len(sorted_features)))
ax1.set_yticklabels(sorted_features[::-1], fontsize=9)
ax1.set_xlabel('Importance')
ax1.set_title(f'Random Forest Feature Importance\n(R²={r2:.3f})')

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2ecc71', label='Currently in model'),
                   Patch(facecolor='#3498db', label='Currently commented out')]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8)

# Plot 2: Correlation with error
ax2 = axes[1]
colors2 = ['#e74c3c' if v > 0 else '#3498db' for v in corr_with_error.values]
ax2.barh(range(len(corr_with_error)), corr_with_error.values[::-1], color=colors2[::-1])
ax2.set_yticks(range(len(corr_with_error)))
ax2.set_yticklabels(corr_with_error.index[::-1], fontsize=9)
ax2.set_xlabel('Pearson Correlation with error_m')
ax2.set_title('Feature Correlation with Localization Error')
ax2.axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to results/feature_importance.png")

# ============================================================================
# INTER-FEATURE CORRELATION (flag redundant pairs)
# ============================================================================

print("\nHighly correlated feature pairs (|r| > 0.85, potential redundancy):")
corr_matrix = df_clean[available_features].corr()
found = False
for i in range(len(available_features)):
    for j in range(i+1, len(available_features)):
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.85:
            print(f"  {available_features[i]} <-> {available_features[j]}: {val:.3f}")
            found = True
if not found:
    print("  None found above threshold")