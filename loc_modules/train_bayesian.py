"""
Hierarchical Bayesian Model for Localization Error Prediction

Learns dataset-specific effects with partial pooling:
- Each dataset gets its own feature weights
- Weights are pulled toward global means (sharing information)
- Naturally handles new datasets (start from global prior)
- Built-in uncertainty quantification

Uses PyMC for inference.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import joblib
import os

print("="*60)
print("HIERARCHICAL BAYESIAN ERROR PREDICTION")
print("="*60)

# ============================================================================
# LOAD DATA
# ============================================================================

# Load all three datasets
print("\nLoading datasets...")
df_fr1 = pd.read_csv("results/fr1_uncertainty_with_geometry.csv")
df_fr3 = pd.read_csv("results/fr3_uncertainty_with_geometry.csv")
df_mh_01 = pd.read_csv("results/mh_01_uncertainty_with_geometry.csv")
df_mh_03 = pd.read_csv("results/mh_03_uncertainty_with_geometry.csv")

df_fr1['dataset'] = 0  # FR1 = 0
df_fr3['dataset'] = 1  # FR3 = 1
df_mh_01['dataset'] = 2  # MH_01 = 2
df_mh_03['dataset'] = 3  # MH_03 = 3

df_combined = pd.concat([df_fr1, df_fr3, df_mh_01, df_mh_03], ignore_index=True)

print(f"  FR1: {len(df_fr1)} frames")
print(f"  FR3: {len(df_fr3)} frames")
print(f"  MH_01: {len(df_mh_01)} frames")
print(f"  MH_03: {len(df_mh_03)} frames")
print(f"  Total: {len(df_combined)} frames")

# ============================================================================
# SELECT FEATURES (Keep it simple - top features from RF)
# ============================================================================

# Use only top 5 most important features from RF analysis
# This keeps the model tractable and interpretable
features = [
    "img_contrast",           # 15%
    "img_brightness",         # 11%
    "mean_inlier_track_length",  # 8%
    "mean_inverse_depth",     # 7%
    "img_histogram_uniformity",  # 6%
    "img_edge_density",       # 4%
    "mean_inlier_ba_error"    # 3%
]

print(f"\nUsing {len(features)} features:")
for f in features:
    print(f"  - {f}")

# Extract data
X = df_combined[features].values
y = df_combined['error_m'].values  # meters
dataset_idx = df_combined['dataset'].values

# Standardize features (helps with convergence)
from sklearn.preprocessing import StandardScaler, RobustScaler

# Use RobustScaler for image quality features (handles outliers better)
# Regular StandardScaler for others
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Additional clipping to remove extreme outliers
for i in range(X_scaled.shape[1]):
    X_scaled[:, i] = np.clip(X_scaled[:, i], -5, 5)  # Clip at ±5 std

# Also standardize y (helps with prior specification)
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

print(f"\nData preparation:")
print(f"  X shape: {X_scaled.shape}")
print(f"  y range (scaled): [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")
print(f"  Datasets: {np.unique(dataset_idx)}")

# ============================================================================
# BUILD HIERARCHICAL MODEL
# ============================================================================

print("\n" + "="*60)
print("BUILDING BAYESIAN MODEL")
print("="*60)

n_features = X_scaled.shape[1]
n_datasets = len(np.unique(dataset_idx))

with pm.Model() as hierarchical_model:
    
    # ========== HYPERPRIORS (Global parameters) ==========
    # These represent the "average" effect across all datasets
    
    # Global mean effect for each feature
    μ_global = pm.Normal('μ_global', mu=0, sigma=1, shape=n_features)
    
    # Global variation in effects across datasets
    # (How much do datasets differ from each other?)
    σ_global = pm.HalfNormal('σ_global', sigma=0.5, shape=n_features)
    
    # ========== DATASET-SPECIFIC EFFECTS (Partial pooling) ==========
    # Each dataset gets its own weights, but pulled toward global mean
    
    # Shape: (n_datasets, n_features)
    β = pm.Normal('β', 
                  mu=μ_global, 
                  sigma=σ_global, 
                  shape=(n_datasets, n_features))
    
    # ========== INTERCEPTS ==========
    # Global intercept
    α_global = pm.Normal('α_global', mu=0, sigma=1)
    
    # Dataset-specific intercepts
    σ_α = pm.HalfNormal('σ_α', sigma=0.5)
    α = pm.Normal('α', mu=α_global, sigma=σ_α, shape=n_datasets)
    
    # ========== OBSERVATION MODEL ==========
    # Noise (measurement uncertainty)
    σ_obs = pm.HalfNormal('σ_obs', sigma=0.5)
    
    # Linear prediction: y = α[dataset] + β[dataset] · X
    # Use pm.math.dot for matrix multiplication
    μ = α[dataset_idx] + pm.math.sum(β[dataset_idx] * X_scaled, axis=1)
    
    # Likelihood
    y_obs = pm.Normal('y_obs', mu=μ, sigma=σ_obs, observed=y_scaled)

print("\nModel structure:")
print(f"  Hyperpriors: μ_global ({n_features}), σ_global ({n_features})")
print(f"  Dataset effects: β ({n_datasets} × {n_features})")
print(f"  Intercepts: α ({n_datasets})")
print(f"  Observation noise: σ_obs")

# ============================================================================
# INFERENCE (Sampling from posterior)
# ============================================================================

print("\n" + "="*60)
print("RUNNING INFERENCE (MCMC)")
print("="*60)
print("This may take a few minutes...")

with hierarchical_model:
    # Use NUTS sampler (No-U-Turn Sampler)
    # draws=2000, tune=1000 is reasonable for initial analysis
    # chains=2 for checking convergence
    trace = pm.sample(
        draws=2000,
        tune=1000, 
        chains=2,
        return_inferencedata=True,
        random_seed=42,
        target_accept=0.9  # Higher acceptance rate for better convergence
    )

print("✓ Sampling complete")

# ============================================================================
# CONVERGENCE DIAGNOSTICS
# ============================================================================

print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

# Check R-hat (should be < 1.01 for good convergence)
summary = az.summary(trace, var_names=['μ_global', 'σ_global', 'β', 'σ_obs'])
print("\nR-hat values (should be < 1.01):")
print(f"  Max R-hat: {summary['r_hat'].max():.4f}")
print(f"  Min ESS (bulk): {summary['ess_bulk'].min():.0f}")

if summary['r_hat'].max() > 1.01:
    print("\n⚠ WARNING: Some parameters have R-hat > 1.01")
    print("   Model may not have converged. Consider:")
    print("   - Increasing draws/tune")
    print("   - Reducing number of features")
else:
    print("✓ Convergence looks good!")

# ============================================================================
# POSTERIOR ANALYSIS
# ============================================================================

print("\n" + "="*60)
print("POSTERIOR ESTIMATES")
print("="*60)

# Extract posterior means
μ_global_post = trace.posterior['μ_global'].mean(dim=['chain', 'draw']).values
σ_global_post = trace.posterior['σ_global'].mean(dim=['chain', 'draw']).values
β_post = trace.posterior['β'].mean(dim=['chain', 'draw']).values  # (n_datasets, n_features)

print("\nGLOBAL FEATURE EFFECTS (averaged across datasets):")
print(f"{'Feature':35s} {'Mean':>8s} {'Std':>8s} {'95% CI':>20s}")
print("-"*75)

for i, feat_name in enumerate(features):
    mean = μ_global_post[i]
    std = σ_global_post[i]
    
    # Get 95% credible interval
    ci_low = np.percentile(trace.posterior['μ_global'].values[:, :, i], 2.5)
    ci_high = np.percentile(trace.posterior['μ_global'].values[:, :, i], 97.5)
    
    # Direction indicator
    direction = "↓ error" if mean < 0 else "↑ error"
    significance = "***" if (ci_low * ci_high) > 0 else " "  # CI doesn't contain 0
    
    print(f"{feat_name:35s} {mean:+8.3f} {std:8.3f} [{ci_low:+7.3f}, {ci_high:+7.3f}] {direction} {significance}")

print("\nDATASET-SPECIFIC EFFECTS:")
dataset_names = ['FR1', 'FR3', 'MH_01', 'MH_03']

for d in range(n_datasets):
    print(f"\n{dataset_names[d]}:")
    print(f"{'Feature':35s} {'Effect':>8s}")
    print("-"*45)
    
    for i, feat_name in enumerate(features):
        effect = β_post[d, i]
        marker = "***" if abs(effect) > 0.5 else ("**" if abs(effect) > 0.3 else "*")
        print(f"{feat_name:35s} {effect:+8.3f}  {marker}")

# ============================================================================
# PREDICTION & VALIDATION
# ============================================================================

print("\n" + "="*60)
print("PREDICTION PERFORMANCE")
print("="*60)

# Posterior predictive (in-sample)
with hierarchical_model:
    post_pred = pm.sample_posterior_predictive(trace, random_seed=42)

y_pred_scaled = post_pred.posterior_predictive['y_obs'].mean(dim=['chain', 'draw']).values

# Unscale predictions
y_pred = y_pred_scaled * y_std + y_mean

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y, y_pred))
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nIn-sample performance:")
print(f"  RMSE: {rmse*100:.2f} cm")
print(f"  MAE:  {mae*100:.2f} cm")
print(f"  R²:   {r2:.3f}")

# Per-dataset performance
for d, name in enumerate(dataset_names):
    mask = dataset_idx == d
    rmse_d = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    r2_d = r2_score(y[mask], y_pred[mask])
    print(f"\n{name}:")
    print(f"  RMSE: {rmse_d*100:.2f} cm")
    print(f"  R²:   {r2_d:.3f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

os.makedirs('results', exist_ok=True)

# Save trace (for later analysis)
trace.to_netcdf('results/bayesian_trace.nc')

# Save model summary
model_data = {
    'features': features,
    'scaler': scaler,
    'y_mean': y_mean,
    'y_std': y_std,
    'μ_global': μ_global_post,
    'σ_global': σ_global_post,
    'β': β_post,
    'dataset_names': dataset_names,
    'rmse': rmse,
    'r2': r2
}

joblib.dump(model_data, 'results/bayesian_model.joblib')
print("\n✓ Saved model to results/bayesian_model.joblib")
print("✓ Saved trace to results/bayesian_trace.nc")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Global effects with uncertainty
ax = axes[0, 0]
x_pos = np.arange(len(features))
ax.errorbar(x_pos, μ_global_post, yerr=2*σ_global_post, fmt='o', capsize=5, capthick=2)
ax.axhline(0, color='red', linestyle='--', alpha=0.5)
ax.set_xticks(x_pos)
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Global Effect (standardized)', fontsize=11)
ax.set_title('Global Feature Effects\n(error bars = ±2σ)', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)

# Plot 2: Dataset comparison
ax = axes[0, 1]
x = np.arange(len(features))
width = 0.25
ax.bar(x - width, β_post[0, :], width, label='FR1', alpha=0.8)
ax.bar(x, β_post[1, :], width, label='FR3', alpha=0.8)
ax.bar(x + width, β_post[2, :], width, label='MH_01', alpha=0.8)
ax.bar(x + 2*width, β_post[3, :], width, label='MH_03', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Effect Size', fontsize=11)
ax.set_title('Dataset-Specific Effects', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)

# Plot 3: Predicted vs Actual
ax = axes[1, 0]
ax.scatter(y * 100, y_pred * 100, alpha=0.5, s=20)
ax.plot([0, y.max()*100], [0, y.max()*100], 'r--', lw=2, label='Perfect')
ax.set_xlabel('Actual Error (cm)', fontsize=11)
ax.set_ylabel('Predicted Error (cm)', fontsize=11)
ax.set_title(f'Predicted vs Actual\nR² = {r2:.3f}, RMSE = {rmse*100:.2f} cm', 
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Posterior distributions for top feature
ax = axes[1, 1]
top_feature_idx = np.argmax(np.abs(μ_global_post))
top_feature_name = features[top_feature_idx]

samples_fr1 = trace.posterior['β'].values[:, :, 0, top_feature_idx].flatten()
samples_fr3 = trace.posterior['β'].values[:, :, 1, top_feature_idx].flatten()

ax.hist(samples_fr1, bins=50, alpha=0.6, label='FR1', density=True)
ax.hist(samples_fr3, bins=50, alpha=0.6, label='FR3', density=True)
ax.axvline(β_post[0, top_feature_idx], color='blue', linestyle='--', lw=2)
ax.axvline(β_post[1, top_feature_idx], color='orange', linestyle='--', lw=2)
ax.set_xlabel(f'Effect of {top_feature_name}', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f'Posterior Distribution\n(Top Feature: {top_feature_name})', 
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/bayesian_model_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/bayesian_model_analysis.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nKey findings:")
print(f"1. Model achieved R² = {r2:.3f} with {len(features)} features")
print(f"2. Global effects show which features matter universally")
print(f"3. Dataset-specific effects show adaptation:")

# Check which features have large between-dataset variation
adaptivity_scores = σ_global_post / (np.abs(μ_global_post) + 0.1)
most_adaptive_idx = np.argmax(adaptivity_scores)
print(f"   - Most adaptive feature: {features[most_adaptive_idx]}")
print(f"     (high variation across datasets relative to mean effect)")

# Check which features are consistent
least_adaptive_idx = np.argmin(adaptivity_scores)
print(f"   - Most consistent feature: {features[least_adaptive_idx]}")
print(f"     (low variation across datasets)")

print("\nNext steps:")
print("  - Examine posterior distributions (results/bayesian_trace.nc)")
print("  - Use trace for uncertainty propagation")
print("  - Predict on new dataset using global priors")

print("\n" + "="*60)
print("DONE - BAYESIAN MODEL")
print("="*60)