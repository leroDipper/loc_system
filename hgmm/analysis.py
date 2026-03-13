"""
Visualize error distributions to determine number of modes.
Uses KDE and GMM to identify natural clusters in the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture

# ============================================================================
# LOAD DATA
# ============================================================================

dfs = []
for seq in ['mh_01', 'mh_03', 'mh_05']:
    for kp in [250, 350, 550]:
        df = pd.read_csv(f'results/{seq}_uncertainty{kp}.csv')
        df['source'] = seq
        dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# Outlier removal (same as training)
mean_e = df['error_m'].mean()
std_e = df['error_m'].std()
df = df[df['error_m'] < mean_e + 3 * std_e].copy()

print(f"Total frames after outlier removal: {len(df)}")

# ============================================================================
# FIT GMM WITH DIFFERENT NUMBER OF COMPONENTS
# ============================================================================

y = df['error_m'].values.reshape(-1, 1)

print("\nGMM BIC scores (lower is better):")
bic_scores = []
n_components_range = range(1, 7)
for n in n_components_range:
    gmm = GaussianMixture(n_components=n, random_state=42, n_init=5)
    gmm.fit(y)
    bic = gmm.bic(y)
    bic_scores.append(bic)
    print(f"  n={n}: BIC={bic:.1f}")

best_n = n_components_range[np.argmin(bic_scores)]
print(f"\nBest number of components: {best_n}")

# ============================================================================
# PLOT
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Overall error distribution with KDE
ax = axes[0, 0]
x_range = np.linspace(0, df['error_m'].max(), 500)
kde = gaussian_kde(df['error_m'].values, bw_method=0.15)
ax.hist(df['error_m'].values * 100, bins=80, density=True, alpha=0.5, color='steelblue', label='Data')
ax.plot(x_range * 100, kde(x_range) / 100, 'k-', lw=2, label='KDE')
ax.set_xlabel('Error (cm)')
ax.set_ylabel('Density')
ax.set_title('Overall Error Distribution')
ax.legend()

# Plot 2: Per-sequence KDE
ax = axes[0, 1]
colors = {'mh_01': '#2ecc71', 'mh_03': '#3498db', 'mh_05': '#e74c3c'}
for seq, color in colors.items():
    data = df[df['source'] == seq]['error_m'].values
    kde_seq = gaussian_kde(data, bw_method=0.2)
    ax.plot(x_range * 100, kde_seq(x_range) / 100, lw=2, color=color, label=seq)
ax.set_xlabel('Error (cm)')
ax.set_ylabel('Density')
ax.set_title('Per-Sequence Error Distribution')
ax.legend()

# Plot 3: BIC scores
ax = axes[1, 0]
ax.plot(list(n_components_range), bic_scores, 'o-', color='steelblue', lw=2, markersize=8)
ax.axvline(x=best_n, color='red', linestyle='--', label=f'Best n={best_n}')
ax.set_xlabel('Number of Gaussian Components')
ax.set_ylabel('BIC Score')
ax.set_title('GMM Model Selection (BIC)\nLower is better')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Best GMM fit overlaid on data
ax = axes[1, 1]
best_gmm = GaussianMixture(n_components=best_n, random_state=42, n_init=5)
best_gmm.fit(y)

ax.hist(df['error_m'].values * 100, bins=80, density=True, alpha=0.5, color='steelblue', label='Data')

# Plot individual Gaussian components
x_plot = x_range.reshape(-1, 1)
colors_gmm = plt.cm.Set1(np.linspace(0, 1, best_n))
for i in range(best_n):
    mean = best_gmm.means_[i, 0]
    std = np.sqrt(best_gmm.covariances_[i, 0, 0])
    weight = best_gmm.weights_[i]
    component = weight * (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mean) / std) ** 2)
    ax.plot(x_range * 100, component / 100, lw=2, color=colors_gmm[i],
            label=f'Component {i+1}: μ={mean*100:.1f}cm, w={weight:.2f}')

ax.set_xlabel('Error (cm)')
ax.set_ylabel('Density')
ax.set_title(f'Best GMM Fit (n={best_n} components)')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('results/error_distribution.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved plot to results/error_distribution.png")

# ============================================================================
# PRINT GMM COMPONENT DETAILS
# ============================================================================

print(f"\nBest GMM ({best_n} components):")
sorted_idx = np.argsort(best_gmm.means_.flatten())
for i in sorted_idx:
    mean = best_gmm.means_[i, 0]
    std = np.sqrt(best_gmm.covariances_[i, 0, 0])
    weight = best_gmm.weights_[i]
    print(f"  Component {i+1}: mean={mean*100:.1f}cm, std={std*100:.1f}cm, weight={weight:.3f}")