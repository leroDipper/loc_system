"""
Train match confidence model with for all datasets - REGRESSION VERSION

Learns: error_magnitude = f(descriptor features, geometric config, map quality)

Input: uncertainty_with_geometry.csv
Output: Random Forest Regressor predicting continuous error
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import os

print("="*60)
print("TRAINING ERROR PREDICTION MODEL (REGRESSION)")
print("="*60)

# Load data
df_fr1 = pd.read_csv("results/fr1_uncertainty_with_geometry.csv")
df_fr3 = pd.read_csv("results/fr3_uncertainty_with_geometry.csv")
df_mh_01 = pd.read_csv("results/mh_01_uncertainty_with_geometry.csv")

df = pd.concat([df_fr1, df_fr3, df_mh_01], ignore_index=True)
print(f"\nLoaded {len(df)} frames")
print(f"Columns: {len(df.columns)}")

# ============================================================================
# TARGET VARIABLE - CONTINUOUS ERROR
# ============================================================================

print(f"\nTarget variable: Localization error (continuous)")
print(f"  Mean error:   {df['error_m'].mean()*100:.2f} cm")
print(f"  Median error: {df['error_m'].median()*100:.2f} cm")
print(f"  Std error:    {df['error_m'].std()*100:.2f} cm")
print(f"  Min error:    {df['error_m'].min()*100:.2f} cm")
print(f"  Max error:    {df['error_m'].max()*100:.2f} cm")

# ============================================================================
# SELECT FEATURES
# ============================================================================

match_features = [
    "mean_ratio_test",
    "mean_n_candidates",
    "n_matches"
]

geometric_features = [
    "n_inliers",
    "match_spread_normalized",
    "match_std_x",
    "match_std_y",
    "depth_mean",
    "depth_std",
    "depth_relative_std",
    "depth_range",
    "n_quadrants_active",
    "quadrant_entropy",
    "mean_inverse_depth",
    "condition_estimate"
]

map_quality_features = [
    "mean_inlier_track_length",
    "mean_inlier_ba_error"
]

image_quality_features = [
    'img_blur_score', 
    'img_brightness',
    'img_contrast', 
    'img_edge_density', 
    'img_histogram_uniformity'
]

all_features = match_features + geometric_features + map_quality_features + image_quality_features

print(f"\nFeature set:")
print(f"  Match features:       {len(match_features)}")
print(f"  Geometric features:   {len(geometric_features)}")
print(f"  Map quality features: {len(map_quality_features)}")
print(f"  Total:                {len(all_features)}")

# Check availability
available_features = [f for f in all_features if f in df.columns]
missing_features = [f for f in all_features if f not in df.columns]

if missing_features:
    print(f"\n⚠ Warning: Missing features: {missing_features}")
    all_features = available_features

# Extract features and target
X = df[all_features].values
y = df["error_m"].values  # Continuous error in meters

# Handle NaN and inf
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

# Clip extreme values
for i in range(X.shape[1]):
    percentile_99 = np.percentile(X[:, i], 99)
    percentile_1 = np.percentile(X[:, i], 1)
    X[:, i] = np.clip(X[:, i], percentile_1, percentile_99)

print(f"\nFeature matrix: {X.shape}")
print(f"Target vector: {y.shape}")

# ============================================================================
# TRAIN MODEL
# ============================================================================

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

# Build pipeline with Random Forest Regressor
print("\nTraining Random Forest Regressor...")
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)
print("✓ Training complete")

# ============================================================================
# EVALUATE
# ============================================================================

# Predictions
y_pred_train = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

# Metrics
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"Train Set:")
print(f"  RMSE: {rmse_train*100:.2f} cm")
print(f"  MAE:  {mae_train*100:.2f} cm")
print(f"  R²:   {r2_train:.3f}")

print(f"\nTest Set:")
print(f"  RMSE: {rmse_test*100:.2f} cm")
print(f"  MAE:  {mae_test*100:.2f} cm")
print(f"  R²:   {r2_test:.3f}")

# Correlation
from scipy.stats import pearsonr
r, p = pearsonr(y_test, y_pred_test)
print(f"\nCorrelation (predicted vs actual):")
print(f"  Pearson r: {r:+.3f}")
print(f"  p-value:   {p:.2e}")

# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

# Get feature importances from Random Forest
rf_model = pipeline.named_steps['regressor']
importances = rf_model.feature_importances_

print(f"\n{'='*60}")
print(f"FEATURE IMPORTANCES (Random Forest)")
print(f"{'='*60}")
print(f"{'Feature':35s} {'Importance':>12s}")
print(f"{'-'*47}")

# Sort by importance
feature_importance = list(zip(all_features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for name, importance in feature_importance:
    marker = "***" if importance > 0.10 else ("**" if importance > 0.05 else "*")
    print(f"{name:35s} {importance:12.4f}  {marker}")

print(f"\n{'='*60}")
print(f"TOP 5 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
for i, (name, importance) in enumerate(feature_importance[:5], 1):
    print(f"{i}. {name:30s} (importance: {importance:.3f})")

# ============================================================================
# QUANTIFY FEATURE EFFECTS
# ============================================================================

print(f"\n{'='*60}")
print(f"FEATURE EFFECT SIZES (approximate)")
print(f"{'='*60}")

# For top features, compute effect size
# (Change in prediction when feature increases by 1 std dev)
scaler = pipeline.named_steps['scaler']

for name, importance in feature_importance[:5]:
    feat_idx = all_features.index(name)
    
    # Get std dev of this feature (before scaling)
    feat_std = np.std(X[:, feat_idx])
    
    # Create two datasets: median and median+1std
    X_median = np.median(X, axis=0).reshape(1, -1)
    X_plus_std = X_median.copy()
    X_plus_std[0, feat_idx] += feat_std
    
    # Predict
    pred_median = pipeline.predict(X_median)[0]
    pred_plus = pipeline.predict(X_plus_std)[0]
    
    effect = (pred_plus - pred_median) * 100  # cm
    
    direction = "increases" if effect > 0 else "decreases"
    print(f"{name:30s}: +1 std → error {direction} by {abs(effect):.2f} cm")

# ============================================================================
# SAVE MODEL
# ============================================================================

os.makedirs('results', exist_ok=True)

model_data = {
    "model": pipeline,
    "features": all_features,
    "rmse_train": rmse_train,
    "rmse_test": rmse_test,
    "r2_test": r2_test,
    "feature_importance": feature_importance
}

joblib.dump(model_data, "results/error_prediction_model.joblib")
print(f"\n✓ Saved model to results/error_prediction_model.joblib")

# ============================================================================
# VISUALIZE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Predicted vs Actual (test set)
axes[0, 0].scatter(y_test * 100, y_pred_test * 100, alpha=0.5, s=20)
axes[0, 0].plot([0, y_test.max()*100], [0, y_test.max()*100], 'r--', lw=2, label='Perfect prediction')
axes[0, 0].set_xlabel('Actual Error (cm)', fontsize=11)
axes[0, 0].set_ylabel('Predicted Error (cm)', fontsize=11)
axes[0, 0].set_title(f'Predicted vs Actual (Test Set)\nR² = {r2_test:.3f}, RMSE = {rmse_test*100:.2f} cm', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Residuals
residuals = (y_test - y_pred_test) * 100
axes[0, 1].scatter(y_pred_test * 100, residuals, alpha=0.5, s=20)
axes[0, 1].axhline(0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Error (cm)', fontsize=11)
axes[0, 1].set_ylabel('Residual (cm)', fontsize=11)
axes[0, 1].set_title('Residual Plot', fontsize=12)
axes[0, 1].grid(alpha=0.3)

# Plot 3: Feature importances
top_n = min(10, len(importances))
axes[1, 0].barh(range(top_n), [imp for _, imp in feature_importance[:top_n]])
axes[1, 0].set_yticks(range(top_n))
axes[1, 0].set_yticklabels([name for name, _ in feature_importance[:top_n]], fontsize=9)
axes[1, 0].set_xlabel('Importance', fontsize=11)
axes[1, 0].set_title('Top 10 Feature Importances', fontsize=12)
axes[1, 0].grid(axis='x', alpha=0.3)
axes[1, 0].invert_yaxis()

# Plot 4: Error distribution
axes[1, 1].hist(y_test * 100, bins=30, alpha=0.6, label='Actual', color='blue')
axes[1, 1].hist(y_pred_test * 100, bins=30, alpha=0.6, label='Predicted', color='orange')
axes[1, 1].set_xlabel('Error (cm)', fontsize=11)
axes[1, 1].set_ylabel('Count', fontsize=11)
axes[1, 1].set_title('Error Distribution', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results/error_prediction_model.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/error_prediction_model.png")

plt.show()

print(f"\n{'='*60}")
print("DONE - REGRESSION MODEL")
print(f"{'='*60}")