"""
Hierarchical Bayesian Model for Localization Error Prediction
EUROC ONLY WITH PROPER TRAIN/TEST SPLIT

Splits MH_01 into train/test, trains on MH_01_train + MH_03, tests on MH_01_test.
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split

print("="*60)
print("BAYESIAN ERROR PREDICTION - WITH TRAIN/TEST SPLIT")
print("="*60)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\nLoading EuRoC datasets (MH_01 + MH_03)...")

df_mh_01_1 = pd.read_csv("results/mh_01_uncertainty.csv")
df_mh_01_2 = pd.read_csv("results/mh_01_uncertainty350.csv")
df_mh_01_3 = pd.read_csv("results/mh_01_uncertainty550.csv")

df_mh_03_1 = pd.read_csv("results/mh_03_uncertainty.csv")
df_mh_03_2 = pd.read_csv("results/mh_03_uncertainty350.csv")
df_mh_03_3 = pd.read_csv("results/mh_03_uncertainty550.csv")

# ============================================================================
# SPLIT MH_01 INTO TRAIN/TEST (75/25 split by unique frames)
# ============================================================================

print("\nSplitting MH_01 into train/test sets...")

# Get unique frame names from the 350-keypoint version
unique_frames = df_mh_01_2['frame'].unique()
n_total = len(unique_frames)

# 75/25 split
train_frames, test_frames = train_test_split(
    unique_frames, 
    test_size=0.25, 
    random_state=42
)

print(f"  Total unique frames: {n_total}")
print(f"  Train frames: {len(train_frames)} (75%)")
print(f"  Test frames: {len(test_frames)} (25%)")

# Filter each keypoint version
def split_by_frames(df, train_frames, test_frames):
    df_train = df[df['frame'].isin(train_frames)].copy()
    df_test = df[df['frame'].isin(test_frames)].copy()
    return df_train, df_test

df_mh_01_1_train, df_mh_01_1_test = split_by_frames(df_mh_01_1, train_frames, test_frames)
df_mh_01_2_train, df_mh_01_2_test = split_by_frames(df_mh_01_2, train_frames, test_frames)
df_mh_01_3_train, df_mh_01_3_test = split_by_frames(df_mh_01_3, train_frames, test_frames)

# Combine keypoint versions for train and test
df_mh_01_train = pd.concat([df_mh_01_1_train, df_mh_01_2_train, df_mh_01_3_train], ignore_index=True)
df_mh_01_test = pd.concat([df_mh_01_2_test], ignore_index=True)  # Only 350 kpts for test

# Combine MH_03 (all keypoints)
df_mh_03 = pd.concat([df_mh_03_1, df_mh_03_2, df_mh_03_3], ignore_index=True)

print(f"\nTraining data:")
print(f"  MH_01 train: {len(df_mh_01_train)} frames (mean: {df_mh_01_train['error_m'].mean()*100:.2f} cm)")
print(f"  MH_03:       {len(df_mh_03)} frames (mean: {df_mh_03['error_m'].mean()*100:.2f} cm)")

print(f"\nTest data:")
print(f"  MH_01 test:  {len(df_mh_01_test)} frames (mean: {df_mh_01_test['error_m'].mean()*100:.2f} cm)")

# Save test frame names for later
os.makedirs('results', exist_ok=True)
pd.DataFrame({'frame': test_frames}).to_csv('results/mh_01_test_frames.csv', index=False)
print("\n Saved test frame list to results/mh_01_test_frames.csv")

# ============================================================================
# COMBINE TRAINING DATA
# ============================================================================

df_mh_01_train['dataset'] = 0
df_mh_03['dataset'] = 1

df_train = pd.concat([df_mh_01_train, df_mh_03], ignore_index=True)
print(f"\nTotal training: {len(df_train)} frames (mean: {df_train['error_m'].mean()*100:.2f} cm)")

# ============================================================================
# SELECT FEATURES
# ============================================================================

features = [
    #"mean_inverse_depth",
    "n_matches",
    #"depth_mean",
    "depth_range",
    #"match_spread_normalized",
    "match_std_x",
    #"quadrant_entropy",
    "mean_inlier_track_length"
    "n_inliers",
    #"depth_relative_std",
    "img_blur_score",
    #"img_brightness",
    #"img_contrast",
    #"img_edge_density",
    "img_histogram_uniformity",
    "mean_inlier_ba_error"
]

# features = [
#     "n_matches",
#     "n_inliers", 
#     #"match_spread_normalized",
#     "match_std_x",
#     #"match_std_y",
#     "depth_mean",
#     #"depth_std",
#     #"depth_relative_std",
#     #"depth_range",
#     #"n_quadrants_active",
#     "quadrant_entropy",
#     "mean_inverse_depth",
#     #"condition_estimate",
#     "img_blur_score",
#     #"img_brightness",
#     "img_contrast",
#     "img_edge_density",
#     "img_histogram_uniformity"
#     # NO mean_ratio_test - not available with fast match()
# ]

print(f"\nUsing {len(features)} features:")
for f in features:
    print(f"  - {f}")

# ============================================================================
# OUTLIER FILTERING
# ============================================================================

X = df_train[features].values
y = df_train["error_m"].values
dataset_idx = df_train['dataset'].values

# 3-sigma rule
mean_error = y.mean()
std_error = y.std()
threshold = mean_error + 3 * std_error

print(f"\nOutlier filtering (3-sigma rule, threshold={threshold*100:.2f}cm):")
print(f"  Before: {len(y)} frames, mean={mean_error*100:.2f}cm, max={y.max()*100:.2f}cm")

outlier_mask = y < threshold
X = X[outlier_mask]
y = y[outlier_mask]
dataset_idx = dataset_idx[outlier_mask]

print(f"  After: {len(y)} frames, mean={y.mean()*100:.2f}cm, max={y.max()*100:.2f}cm")
print(f"  Removed: {np.sum(~outlier_mask)} extreme outliers")

# ============================================================================
# STANDARDIZE FEATURES
# ============================================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clip extreme values
for i in range(X_scaled.shape[1]):
    X_scaled[:, i] = np.clip(X_scaled[:, i], -5, 5)

# Standardize y
y_mean = y.mean()
y_std = y.std()
y_scaled = (y - y_mean) / y_std

print(f"\nData preparation:")
print(f"  X shape: {X_scaled.shape}")
print(f"  y_mean: {y_mean*100:.2f} cm")
print(f"  y_std: {y_std*100:.2f} cm")

# ============================================================================
# BUILD BAYESIAN MODEL
# ============================================================================

print("\n" + "="*60)
print("BUILDING BAYESIAN MODEL")
print("="*60)

n_features = X_scaled.shape[1]
n_datasets = len(np.unique(dataset_idx))

with pm.Model() as hierarchical_model:

    # ======MIXTURE WEIGHTS============
    beta_mixture = pm.Normal('beta_mixture', mu=0, sigma=1, shape=n_features)
    alpha_mixture = pm.Normal('alpha_mixture', mu=-2, sigma=1)

    #Probaility of failure for each frame
    logit_p = alpha_mixture + pm.math.dot(X_scaled, beta_mixture)
    p_failure = pm.Deterministic('p_failure', pm.math.sigmoid(logit_p))


    # =======NORMAL MODEL============
    # Global parameters
    mu_global = pm.Normal('mu_global', mu=0, sigma=1, shape=n_features)
    sigma_global = pm.HalfNormal('sigma_global', sigma=0.5, shape=n_features)
    
    # Dataset-specific effects 
    beta_offset = pm.Normal('beta_offset', mu=0, sigma=1, shape=(n_datasets, n_features))
    beta = pm.Deterministic('beta', mu_global + beta_offset * sigma_global)
    
    # Intercepts
    alpha_global = pm.Normal('alpha_global', mu=0, sigma=1)
    sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=0.5)
    alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=n_datasets)
    alpha = pm.Deterministic('alpha', alpha_global + alpha_offset * sigma_alpha)
    
    # Normal mode prediction
    mu_normal = alpha[dataset_idx] + pm.math.sum(beta[dataset_idx] * X_scaled, axis=1)
    sigma_normal = pm.HalfNormal('sigma_normal', sigma=0.5)


    # =======FAILURE MODEL============
    # Global feature weights (shared across datasets)
    beta_failure = pm.Normal('beta_failure', mu=0, sigma=1, shape=n_features)

    # Dataset-specific intercepts (each dataset has different baseline failure error)
    alpha_failure_global = pm.Normal('alpha_failure_global', mu=2.0, sigma=0.5)
    alpha_failure_offset = pm.Normal('alpha_failure_offset', mu=0, sigma=0.3, shape=n_datasets)
    alpha_failure = pm.Deterministic('alpha_failure', alpha_failure_global + alpha_failure_offset * 0.5)

    # Failure mode prediction
    mu_failure = alpha_failure[dataset_idx] + pm.math.sum(beta_failure * X_scaled, axis=1)
    sigma_failure = pm.HalfNormal('sigma_failure', sigma=1.0)
    

    # =======MIXTURE===========
    mixture_weights = pm.math.stack([1 - p_failure, p_failure], axis=1)
    # Stack mu parameters: shape (n_observations, n_components)
    mu_stacked = pm.math.stack([mu_normal, mu_failure], axis=1)
    y_obs = pm.NormalMixture('y_obs', w=mixture_weights, mu=mu_stacked, sigma=[sigma_normal, sigma_failure], observed=y_scaled)

# ============================================================================
# INFERENCE
# ============================================================================

print("\n" + "="*60)
print("RUNNING INFERENCE")
print("="*60)

with hierarchical_model:
    trace = pm.sample(
        draws=2000,
        tune=2000, 
        chains=4,
        return_inferencedata=True,
        random_seed=42,
        target_accept=0.95
    )

print("Sampling complete")

# ============================================================================
# CONVERGENCE CHECK
# ============================================================================

print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

summary = az.summary(trace, var_names=['mu_global', 'sigma_global', 'beta', 'sigma_normal', 'sigma_failure'])
print(f"Max R-hat: {summary['r_hat'].max():.4f}")
print(f"Min ESS (bulk): {summary['ess_bulk'].min():.0f}")

if summary['r_hat'].max() > 1.01:
    print("WARNING: Convergence issues detected")
else:
    print("Convergence looks good!")

# ============================================================================
# EXTRACT POSTERIORS
# ============================================================================

mu_global_post = trace.posterior['mu_global'].mean(dim=['chain', 'draw']).values
sigma_global_post = trace.posterior['sigma_global'].mean(dim=['chain', 'draw']).values
beta_post = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
alpha_post = trace.posterior['alpha'].mean(dim=['chain', 'draw']).values



beta_mixture_post = trace.posterior['beta_mixture'].mean(dim=['chain', 'draw']).values
alpha_mixture_post = trace.posterior['alpha_mixture'].mean(dim=['chain', 'draw']).values
beta_failure_post = trace.posterior['beta_failure'].mean(dim=['chain', 'draw']).values
alpha_failure_post = trace.posterior['alpha_failure'].mean(dim=['chain', 'draw']).values
sigma_normal_post = trace.posterior['sigma_normal'].mean(dim=['chain', 'draw']).values
sigma_failure_post = trace.posterior['sigma_failure'].mean(dim=['chain', 'draw']).values

# ============================================================================
# SAVE MODEL
# ============================================================================

model_data = {
    'features': features,
    'scaler': scaler,
    'y_mean': y_mean,
    'y_std': y_std,
    
    # Normal mode
    'mu_global': mu_global_post,
    'sigma_global': sigma_global_post,
    'beta': beta_post,
    'alpha': alpha_post,  
    'sigma_normal': sigma_normal_post,
    
    # Failure mode
    'beta_failure': beta_failure_post,
    'alpha_failure': alpha_failure_post,
    'sigma_failure': sigma_failure_post,
    
    # Mixture weights
    'beta_mixture': beta_mixture_post,
    'alpha_mixture': alpha_mixture_post,
    
    'dataset_names': ['MH_01', 'MH_03'],
    'train_frames': train_frames.tolist(),
    'test_frames': test_frames.tolist(),
    'model_type': 'mixture'  
}

joblib.dump(model_data, 'results/bayesian_model_euroc_mixture.joblib')
trace.to_netcdf('results/bayesian_trace_euroc.nc')
print("\n Saved model to results/bayesian_model_euroc.joblib")

# ============================================================================
# EVALUATE ON TEST SET
# ============================================================================

print("\n" + "="*60)
print("EVALUATING ON HELD-OUT TEST SET")
print("="*60)

# Prepare test data
X_test = df_mh_01_test[features].values
y_test = df_mh_01_test["error_m"].values

# Standardize test features
X_test_scaled = scaler.transform(X_test)
X_test_scaled = np.clip(X_test_scaled, -5, 5)

# Predict failure probability
logit_p_test = alpha_mixture_post + np.dot(X_test_scaled, beta_mixture_post)
p_failure_test = 1 / (1 + np.exp(-logit_p_test))

# Predict normal mode error
mu_normal_test = alpha_post[0] + np.dot(X_test_scaled, beta_post[0]) 
 
# Using MH_01 params (dataset 0)
pred_normal = mu_normal_test * y_std + y_mean

# Predict failure mode error  
mu_failure_test = alpha_failure_post[0] + np.dot(X_test_scaled, beta_failure_post)
pred_failure = mu_failure_test * y_std + y_mean

# Expected error (mixture)
predictions = (1 - p_failure_test) * pred_normal + p_failure_test * pred_failure

# Uncertainty (mixture of variances)
var_normal = sigma_normal_post**2 * (y_std**2)
var_failure = sigma_failure_post**2 * (y_std**2)
uncertainties = np.sqrt(
    (1 - p_failure_test) * (var_normal + pred_normal**2) + 
    p_failure_test * (var_failure + pred_failure**2) - 
    predictions**2
)

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
correlation = np.corrcoef(y_test, predictions)[0, 1]

# Calibration: within 1σ
residuals = np.abs(y_test - predictions)
within_1sigma = np.sum(residuals < uncertainties) / len(residuals)


print(f"\nMixture Model Analysis:")
print(f"  Mean P(failure): {p_failure_test.mean():.3f}")
print(f"  Frames with P(failure) > 0.5: {np.sum(p_failure_test > 0.5)}/{len(p_failure_test)}")

# Stratify by predicted mode
normal_mask = p_failure_test < 0.5
failure_mask = p_failure_test >= 0.5

if np.sum(normal_mask) > 0:
    print(f"\n  Predicted Normal Mode (n={np.sum(normal_mask)}):")
    print(f"    Mean actual error: {y_test[normal_mask].mean()*100:.2f} cm")
    
if np.sum(failure_mask) > 0:
    print(f"\n  Predicted Failure Mode (n={np.sum(failure_mask)}):")
    print(f"    Mean actual error: {y_test[failure_mask].mean()*100:.2f} cm")

print(f"\nTest Set Performance:")
print(f"  Frames: {len(y_test)}")
print(f"  Mean actual error: {y_test.mean()*100:.2f} cm")
print(f"  Mean predicted error: {predictions.mean()*100:.2f} cm")
print(f"  Mean predicted uncertainty: {uncertainties.mean()*100:.2f} cm")
print(f"\nMetrics:")
print(f"  RMSE: {rmse*100:.2f} cm")
print(f"  MAE:  {mae*100:.2f} cm")
print(f"  R²:   {r2:.3f}")
print(f"  Correlation: {correlation:.3f}")
print(f"\nCalibration:")
print(f"  Within sigma: {within_1sigma*100:.1f}% (expected ~68%)")
print(f"  Mean calibration error: {mae*100:.2f} cm")

# Reliability bins
print(f"\nReliability by Predicted Uncertainty:")
bins = [(0, 5), (5, 10), (10, 15), (15, 100)]
for low, high in bins:
    mask = (uncertainties*100 >= low) & (uncertainties*100 < high)
    if np.sum(mask) > 0:
        actual = y_test[mask] * 100
        print(f"  {low:2d}-{high:2d}cm predicted: n={np.sum(mask):3d}, actual={actual.mean():.2f}±{actual.std():.2f}cm")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nTraining:")
print(f"  Trained on {len(train_frames)} unique MH_01 frames + all MH_03")
print(f"  Training mean error: {y_mean*100:.2f} cm")

print(f"\nTesting (Held-Out):")
print(f"  Tested on {len(test_frames)} held-out MH_01 frames")
print(f"  Correlation: {correlation:.3f}")
print(f"  Within sigma: {within_1sigma*100:.1f}%")
print(f"  Calibration error: {mae*100:.2f} cm")

if correlation > 0.40 and within_1sigma > 0.65:
    print("\n Model shows good generalization to held-out data!")
elif correlation > 0.35:
    print("\n Model shows moderate generalization")
else:
    print("\n✗ Model shows limited generalization")

print("\n" + "="*60)
print("DONE")
print("="*60)