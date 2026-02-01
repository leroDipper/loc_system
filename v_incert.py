"""
Visualize uncertainty estimation results.

Plots:
1. Localization error vs predicted uncertainty
2. Confidence distribution
3. Uncertainty over time
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os

# Load uncertainty data from CSV
if not os.path.exists('results/uncertainty_estimates.csv'):
    print("Error: results/uncertainty_estimates.csv not found.")
    print("Run main_tumfr1_with_uncertainty.py first.")
    exit()

df = pd.read_csv('results/uncertainty_estimates.csv')

print("="*60)
print("UNCERTAINTY ANALYSIS")
print("="*60)
print(f"Frames analyzed: {len(df)}")
print(f"\nData columns: {list(df.columns)}")

# Extract key metrics
errors = df['error_m'].values
confidences = df['mean_confidence'].values
pixel_uncertainties = df['mean_pixel_uncertainty_px'].values
trans_uncertainty = df['translation_std_total_m'].values

# Correlation between predicted uncertainty and actual error
corr, p_value = pearsonr(trans_uncertainty, errors)
print(f"\nCorrelation (uncertainty vs error):")
print(f"  Pearson r: {corr:.3f}")
print(f"  p-value:   {p_value:.3e}")
if p_value < 0.05:
    print(f"  ✓ Significant correlation!")
else:
    print(f"  ✗ Not significant")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Error vs Predicted Uncertainty
ax = axes[0, 0]
ax.scatter(trans_uncertainty * 1000, errors * 100, alpha=0.5, s=20)
ax.set_xlabel('Predicted Translation Uncertainty (mm)')
ax.set_ylabel('Actual Localization Error (cm)')
ax.set_title('Predicted Uncertainty vs Actual Error')
ax.grid(alpha=0.3)

# Add trend line
if len(trans_uncertainty) > 10:
    z = np.polyfit(trans_uncertainty * 1000, errors * 100, 1)
    p = np.poly1d(z)
    x_line = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Fit: y={z[0]:.2f}x+{z[1]:.2f}')
    ax.legend()

# Plot 2: Confidence Distribution
ax = axes[0, 1]
ax.hist(confidences, bins=30, alpha=0.7, color='green', edgecolor='black')
ax.set_xlabel('Mean Match Confidence')
ax.set_ylabel('Count')
ax.set_title('Match Confidence Distribution')
ax.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Uncertainty over time
ax = axes[1, 0]
frame_indices = np.arange(len(trans_uncertainty))
ax.plot(frame_indices, trans_uncertainty * 1000, alpha=0.7, linewidth=1)
ax.set_xlabel('Frame Index')
ax.set_ylabel('Translation Uncertainty (mm)')
ax.set_title('Uncertainty Over Time')
ax.grid(alpha=0.3)

# Plot 4: Pixel Uncertainty vs Match Confidence
ax = axes[1, 1]
ax.scatter(confidences, pixel_uncertainties, alpha=0.5, s=20)
ax.set_xlabel('Mean Match Confidence')
ax.set_ylabel('Mean Pixel Uncertainty (px)')
ax.set_title('Pixel Uncertainty vs Confidence')
ax.grid(alpha=0.3)

# Add expected relationship line (σ = 1.0 / sqrt(confidence))
conf_range = np.linspace(0.3, 1.0, 100)
expected_sigma = 1.0 / np.sqrt(conf_range)
ax.plot(conf_range, expected_sigma, 'r--', linewidth=2, label='Expected: σ=1/√p', alpha=0.8)
ax.legend()

plt.tight_layout()
plt.savefig('results/uncertainty_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to results/uncertainty_analysis.png")

# Print summary statistics
print(f"\n{'='*60}")
print("SUMMARY STATISTICS")
print(f"{'='*60}")
print(f"\nMatch Confidence:")
print(f"  Mean:   {confidences.mean():.3f}")
print(f"  Median: {np.median(confidences):.3f}")
print(f"  Std:    {confidences.std():.3f}")
print(f"  Min:    {confidences.min():.3f}")
print(f"  Max:    {confidences.max():.3f}")

print(f"\nPixel Uncertainty:")
print(f"  Mean:   {pixel_uncertainties.mean():.2f} px")
print(f"  Median: {np.median(pixel_uncertainties):.2f} px")
print(f"  Std:    {pixel_uncertainties.std():.2f} px")

print(f"\nTranslation Uncertainty:")
print(f"  Mean:   {trans_uncertainty.mean()*1000:.2f} mm")
print(f"  Median: {np.median(trans_uncertainty)*1000:.2f} mm")
print(f"  Std:    {trans_uncertainty.std()*1000:.2f} mm")

print(f"\nLocalization Error:")
print(f"  Mean:   {errors.mean()*100:.2f} cm")
print(f"  Median: {np.median(errors)*100:.2f} cm")
print(f"  Std:    {errors.std()*100:.2f} cm")

print(f"\n{'='*60}")

# Show sample of the CSV data
print("\nSample of CSV data (first 5 rows):")
print(df.head().to_string())