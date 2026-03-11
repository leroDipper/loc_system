"""
Hierarchical Bayesian Model for Localisation Error Prediction
EUROC ONLY WITH PROPER TRAIN/TEST SPLIT

Splits MH_01 into train/test, trains on MH_01_train + MH_03 + MH_05, tests on MH_01_test.
Keypoint versions: 250, 350, 550 only (100 excluded — not representative of deployment at 250 kpts).
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
print("BAYESIAN ERROR PREDICTION")
print("="*60)


# LOAD DATA


print("\nLoading EuRoC datasets ...")

df_mh_01_1 = pd.read_csv("results/mh_01_uncertainty250.csv")
df_mh_01_2 = pd.read_csv("results/mh_01_uncertainty350.csv")
df_mh_01_3 = pd.read_csv("results/mh_01_uncertainty550.csv")

df_mh_03_1 = pd.read_csv("results/mh_03_uncertainty250.csv")
df_mh_03_2 = pd.read_csv("results/mh_03_uncertainty350.csv")
df_mh_03_3 = pd.read_csv("results/mh_03_uncertainty550.csv")

df_mh_05_1 = pd.read_csv("results/mh_05_uncertainty250.csv")
df_mh_05_2 = pd.read_csv("results/mh_05_uncertainty350.csv")
df_mh_05_3 = pd.read_csv("results/mh_05_uncertainty550.csv")


# SPLIT MH_01 INTO TRAIN/TEST (75/25 split by unique frames)

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
df_mh_01_test = pd.concat([df_mh_01_1_test], ignore_index=True)  # Only 250 kpts for test

# Combine MH_03 (250, 350, 550) — split 75/25
unique_frames_03 = df_mh_03_2['frame'].unique()
train_frames_03, test_frames_03 = train_test_split(unique_frames_03, test_size=0.25, random_state=42)
df_mh_03_train = pd.concat([df_mh_03_1, df_mh_03_2, df_mh_03_3], ignore_index=True)
df_mh_03_train = df_mh_03_train[df_mh_03_train['frame'].isin(train_frames_03)].copy()
df_mh_03_test = df_mh_03_1[df_mh_03_1['frame'].isin(test_frames_03)].copy()  # 250 kpts only for test

# Combine MH_05 (250, 350, 550) — split 75/25
unique_frames_05 = df_mh_05_2['frame'].unique()
train_frames_05, test_frames_05 = train_test_split(unique_frames_05, test_size=0.25, random_state=42)
df_mh_05_train = pd.concat([df_mh_05_1, df_mh_05_2, df_mh_05_3], ignore_index=True)
df_mh_05_train = df_mh_05_train[df_mh_05_train['frame'].isin(train_frames_05)].copy()
df_mh_05_test = df_mh_05_1[df_mh_05_1['frame'].isin(test_frames_05)].copy()  # 250 kpts only for test

print(f"\nTraining data:")
print(f"  MH_01 train: {len(df_mh_01_train)} frames (mean: {df_mh_01_train['error_m'].mean()*100:.2f} cm)")
print(f"  MH_03 train: {len(df_mh_03_train)} frames (mean: {df_mh_03_train['error_m'].mean()*100:.2f} cm)")
print(f"  MH_05 train: {len(df_mh_05_train)} frames (mean: {df_mh_05_train['error_m'].mean()*100:.2f} cm)")

print(f"\nTest data:")
print(f"  MH_01 test:  {len(df_mh_01_test)} frames (mean: {df_mh_01_test['error_m'].mean()*100:.2f} cm)")
print(f"  MH_03 test:  {len(df_mh_03_test)} frames (mean: {df_mh_03_test['error_m'].mean()*100:.2f} cm)")
print(f"  MH_05 test:  {len(df_mh_05_test)} frames (mean: {df_mh_05_test['error_m'].mean()*100:.2f} cm)")

# Save test frame names for later
os.makedirs('results', exist_ok=True)
pd.DataFrame({'frame': test_frames}).to_csv('results/mh_01_test_frames.csv', index=False)
print("\n Saved test frame list to results/mh_01_test_frames.csv")


# COMBINE TRAINING DATA


df_mh_01_train['dataset'] = 0
df_mh_03_train['dataset'] = 1
df_mh_05_train['dataset'] = 2

df_train = pd.concat([df_mh_01_train, df_mh_03_train, df_mh_05_train], ignore_index=True)
print(f"\nTotal training: {len(df_train)} frames (mean: {df_train['error_m'].mean()*100:.2f} cm)")


# LOG TRANSFORMS


for df in [df_train, df_mh_01_test, df_mh_03_test, df_mh_05_test]:
    df['log_mean_inlier_track_length'] = np.log1p(df['mean_inlier_track_length'])
    df['log_mean_inverse_depth'] = np.log1p(df['mean_inverse_depth'])
    df['log_median_inlier_reproj_error'] = np.log1p(df['median_inlier_reproj_error'])
    df['log_img_blur_score'] = np.log1p(df['img_blur_score'])


# SELECT FEATURES

features = [
    "log_mean_inlier_track_length",
    "condition_estimate",
    "match_std_x",
    "depth_mean",
    "depth_std",
    "img_contrast",
    "depth_range",
    "img_brightness",
    "match_spread_normalized",
    "log_mean_inverse_depth",
    "log_median_inlier_reproj_error",
    "log_img_blur_score",
    "mean_inlier_ba_error",
]

print(f"\nUsing {len(features)} features:")
for f in features:
    print(f"  - {f}")


# OUTLIER FILTERING


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


# STANDARDISE FEATURES


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clip extreme values
for i in range(X_scaled.shape[1]):
    X_scaled[:, i] = np.clip(X_scaled[:, i], -5, 5)



from scipy.stats import iqr
y_mean = np.median(y)
y_std = iqr(y) / 1.349  
y_scaled = (y - y_mean) / y_std



# BUILD BAYESIAN MODEL


print("\n" + "="*60)
print("BUILDING BAYESIAN MODEL")
print("="*60)

n_features = X_scaled.shape[1]
n_datasets = len(np.unique(dataset_idx))

with pm.Model() as hierarchical_model:
    mode_num = 3

    # MIXTURE WEIGHTS
    beta_mixture = pm.Normal('beta_mixture', mu=0, sigma=1, shape=(n_features, mode_num))

    # Dataset-specific baseline per component
    alpha_mixture_global = pm.Normal('alpha_mixture_global', mu=0, sigma=1, shape=mode_num)
    sigma_alpha_mixture = pm.HalfNormal('sigma_alpha_mixture', sigma=0.5)
    alpha_mixture_offset = pm.Normal('alpha_mixture_offset', mu=0, sigma=1, shape=(n_datasets, mode_num))
    alpha_mixture = pm.Deterministic('alpha_mixture', alpha_mixture_global + alpha_mixture_offset * sigma_alpha_mixture)

    # Softmax mixture weights
    logits = alpha_mixture[dataset_idx] + pm.math.dot(X_scaled, beta_mixture)
    w = pm.Deterministic('w', pm.math.softmax(logits, axis=1))

    # MODE 1 (low error)
        # Global feature weights — shared across datasets
    beta_1 = pm.Normal('beta_1', mu=0, sigma=1, shape=n_features)

        # Dataset-specific intercepts
    alpha_1_global = pm.Normal('alpha_1_global', mu=-1, sigma=0.5)
    sigma_1_alpha = pm.HalfNormal('sigma_1_alpha', sigma=0.5)
    alpha_1_offset = pm.Normal('alpha_1_offset', mu=0, sigma=1, shape=n_datasets)
    alpha_1 = pm.Deterministic('alpha_1', alpha_1_global + alpha_1_offset * sigma_1_alpha)

        # Mode prediction
    mu_1 = alpha_1[dataset_idx] + pm.math.dot(X_scaled, beta_1)
    sigma_1 = pm.HalfNormal('sigma_1', sigma=0.5)

    # MODE 2 (moderate error)
        # Global feature weights — shared across datasets
    beta_2 = pm.Normal('beta_2', mu=0, sigma=1, shape=n_features)

        # Ordering constraint — alpha_2 must be above alpha_1
    alpha_gap_21 = pm.HalfNormal('alpha_gap_21', sigma=1.0)
    alpha_2_global = pm.Deterministic('alpha_2_global', alpha_1_global + alpha_gap_21)

        # Dataset-specific intercepts
    sigma_2_alpha = pm.HalfNormal('sigma_2_alpha', sigma=0.5)
    alpha_2_offset = pm.Normal('alpha_2_offset', mu=0, sigma=1, shape=n_datasets)
    alpha_2 = pm.Deterministic('alpha_2', alpha_2_global + alpha_2_offset * sigma_2_alpha)

        # Mode prediction
    mu_2 = alpha_2[dataset_idx] + pm.math.dot(X_scaled, beta_2)
    sigma_2 = pm.HalfNormal('sigma_2', sigma=0.5)

    # MODE 3 (failure)
        # Global feature weights — shared across datasets
    beta_3 = pm.Normal('beta_3', mu=0, sigma=1, shape=n_features)

        # Ordering constraint — alpha_3 must be above alpha_2
    alpha_gap_32 = pm.HalfNormal('alpha_gap_32', sigma=1.0)
    alpha_3_global = pm.Deterministic('alpha_3_global', alpha_2_global + alpha_gap_32)

        # Dataset-specific intercepts
    sigma_3_alpha = pm.HalfNormal('sigma_3_alpha', sigma=0.5)
    alpha_3_offset = pm.Normal('alpha_3_offset', mu=0, sigma=1, shape=n_datasets)
    alpha_3 = pm.Deterministic('alpha_3', alpha_3_global + alpha_3_offset * sigma_3_alpha)

        # Mode prediction
    mu_3 = alpha_3[dataset_idx] + pm.math.dot(X_scaled, beta_3)
    sigma_3 = pm.HalfNormal('sigma_3', sigma=1.0)


    # MIXTURE
    mu_stacked = pm.math.stack([mu_1, mu_2, mu_3], axis=1)
    y_obs = pm.NormalMixture('y_obs', w=w, mu=mu_stacked, sigma=[sigma_1, sigma_2, sigma_3], observed=y_scaled)



# INFERENCE
print("\n" + "="*60)
print("RUNNING INFERENCE (NUTS - single chain)")
print("="*60)

with hierarchical_model:
    trace = pm.sample(
        draws=2000,
        tune=2000,
        chains=1,
        return_inferencedata=True,
        random_seed=42,
        target_accept=0.95,
        nuts_sampler_kwargs={"max_treedepth": 15}
    )

print("Sampling complete")


# CONVERGENCE CHECK


print("\n" + "="*60)
print("CONVERGENCE DIAGNOSTICS")
print("="*60)

# Single chain — use ESS only (R-hat requires multiple chains)
summary = az.summary(trace, var_names=['alpha_1', 'alpha_2', 'alpha_3', 'sigma_1', 'sigma_2', 'sigma_3'])
print(f"Min ESS (bulk): {summary['ess_bulk'].min():.0f}")

if summary['ess_bulk'].min() < 200:
    print("WARNING: Low ESS — consider more draws")
else:
    print("Convergence looks good!")



# EXTRACT POSTERIORS


# Mixture weights
beta_mixture_post = trace.posterior['beta_mixture'].mean(dim=['chain', 'draw']).values  # (n_features, 3)
alpha_mixture_post = trace.posterior['alpha_mixture'].mean(dim=['chain', 'draw']).values  # (n_datasets, 3)

# Mode 1 (low error)
alpha_1_post = trace.posterior['alpha_1'].mean(dim=['chain', 'draw']).values  # (n_datasets,)
beta_1_post = trace.posterior['beta_1'].mean(dim=['chain', 'draw']).values    # (n_features,)
sigma_1_post = trace.posterior['sigma_1'].mean(dim=['chain', 'draw']).values  # scalar

# Mode 2 (moderate error)
alpha_2_post = trace.posterior['alpha_2'].mean(dim=['chain', 'draw']).values
beta_2_post = trace.posterior['beta_2'].mean(dim=['chain', 'draw']).values    # (n_features,)
sigma_2_post = trace.posterior['sigma_2'].mean(dim=['chain', 'draw']).values

# Mode 3 (failure)
alpha_3_post = trace.posterior['alpha_3'].mean(dim=['chain', 'draw']).values
beta_3_post = trace.posterior['beta_3'].mean(dim=['chain', 'draw']).values    # (n_features,)
sigma_3_post = trace.posterior['sigma_3'].mean(dim=['chain', 'draw']).values



# SAVE MODEL

model_data = {
    'features': features,
    'scaler': scaler,
    'y_mean': y_mean,
    'y_std': y_std,

    # Mixture weights
    'beta_mixture': beta_mixture_post,
    'alpha_mixture': alpha_mixture_post,

    # Mode 1
    'alpha_1': alpha_1_post,
    'beta_1': beta_1_post,
    'sigma_1': sigma_1_post,

    # Mode 2
    'alpha_2': alpha_2_post,
    'beta_2': beta_2_post,
    'sigma_2': sigma_2_post,

    # Mode 3
    'alpha_3': alpha_3_post,
    'beta_3': beta_3_post,
    'sigma_3': sigma_3_post,

    'dataset_names': ['MH_01', 'MH_03', 'MH_05'],
    'train_frames': train_frames.tolist(),
    'test_frames': test_frames.tolist(),
    'model_type': 'mixture_3component'
}

trace.to_netcdf('results/bayesian_trace_euroc.nc')
print("\n Saved trace to results/bayesian_trace_euroc.nc")


# EVALUATE ON TEST SET


print("\n" + "="*60)
print("EVALUATING ON HELD-OUT TEST SET")
print("="*60)

# Prepare test data — combine all three held-out sets (250 kpts only)
df_test_all = pd.concat([df_mh_01_test, df_mh_03_test, df_mh_05_test], ignore_index=True)
X_test = df_test_all[features].values
y_test = df_test_all["error_m"].values

# Standardize test features
X_test_scaled = scaler.transform(X_test)
X_test_scaled = np.clip(X_test_scaled, -5, 5)

# Test dataset indices
n_mh01_test = len(df_mh_01_test)
n_mh03_test = len(df_mh_03_test)
n_mh05_test = len(df_mh_05_test)
test_dataset_idx = np.array([0]*n_mh01_test + [1]*n_mh03_test + [2]*n_mh05_test)

# Compute softmax mixture weights
from scipy.special import softmax
logits_test = alpha_mixture_post[test_dataset_idx] + np.dot(X_test_scaled, beta_mixture_post)  # (n_test, 3)
w_test = softmax(logits_test, axis=1)  # (n_test, 3)

# Predict each mode error
mu_1_test = alpha_1_post[test_dataset_idx] + np.dot(X_test_scaled, beta_1_post)
mu_2_test = alpha_2_post[test_dataset_idx] + np.dot(X_test_scaled, beta_2_post)
mu_3_test = alpha_3_post[test_dataset_idx] + np.dot(X_test_scaled, beta_3_post)

# Denormalize
pred_1 = mu_1_test * y_std + y_mean
pred_2 = mu_2_test * y_std + y_mean
pred_3 = mu_3_test * y_std + y_mean

# Expected error (weighted mixture)
predictions = w_test[:, 0] * pred_1 + w_test[:, 1] * pred_2 + w_test[:, 2] * pred_3

# Uncertainty (law of total variance)
var_1 = sigma_1_post**2 * y_std**2
var_2 = sigma_2_post**2 * y_std**2
var_3 = sigma_3_post**2 * y_std**2
uncertainties = np.sqrt(
    w_test[:, 0] * (var_1 + pred_1**2) +
    w_test[:, 1] * (var_2 + pred_2**2) +
    w_test[:, 2] * (var_3 + pred_3**2) -
    predictions**2
)

# p_failure = weight of component 3
p_failure_test = w_test[:, 2]

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
correlation = np.corrcoef(y_test, predictions)[0, 1]

# Calibration: within 1σ
residuals = np.abs(y_test - predictions)
within_1sigma = np.sum(residuals < uncertainties) / len(residuals)

# Dominant mode per frame
dominant_mode = np.argmax(w_test, axis=1)  # 0, 1, or 2

print(f"\nMixture Model Analysis:")
print(f"  Mean P(failure): {p_failure_test.mean():.3f}")
print(f"  Frames with P(failure) > 0.5: {np.sum(p_failure_test > 0.5)}/{len(p_failure_test)}")

for mode_idx, mode_name in enumerate(['Low-error', 'Moderate-error', 'Failure']):
    mask = dominant_mode == mode_idx
    if np.sum(mask) > 0:
        print(f"\n  Predicted {mode_name} Mode (n={np.sum(mask)}):")
        print(f"    Mean actual error: {y_test[mask].mean()*100:.2f} cm")
        print(f"    Mean weight: {w_test[mask, mode_idx].mean():.3f}")

print(f"\nTest Set Performance:")
print(f"  Frames: {len(y_test)}")
print(f"  Mean actual error: {y_test.mean()*100:.2f} cm")
print(f"  Mean predicted error: {predictions.mean()*100:.2f} cm")
print(f"  Mean predicted uncertainty: {uncertainties.mean()*100:.2f} cm")
print(f"\nMetrics:")
print(f"  RMSE: {rmse*100:.2f} cm")
print(f"  MAE:  {mae*100:.2f} cm")
print(f"  R^2:   {r2:.3f}")
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


model_data['r2'] = float(r2)
model_data['correlation'] = float(correlation)
model_data['rmse'] = float(rmse)
model_data['mae'] = float(mae)
joblib.dump(model_data, 'results/bayesian_gmm.joblib')


# SUMMARY

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print(f"\nTraining:")
print(f"  Trained on {len(train_frames)} unique MH_01 frames + all MH_03 + all MH_05")
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