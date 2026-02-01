"""
Train match confidence model with geometric features - Version 3

Learns: P(match is correct | descriptor features, geometric configuration)

Input: uncertainty_with_geometry.csv (from tum_fr1_unc_geometric.py)
Output: Logistic regression model with geometric features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
import joblib
import matplotlib.pyplot as plt
import os

print("="*60)
print("TRAINING MATCH CONFIDENCE MODEL v3 (WITH GEOMETRIC FEATURES)")
print("="*60)

# Load data from localization run with geometric features
df = pd.read_csv("results/uncertainty_with_geometry.csv")
print(f"\nLoaded {len(df)} frames with geometric features")
print(f"Columns: {list(df.columns)}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# We need to create per-frame targets and features
# For training, we'll consider frames with low error as "good" configurations
# and frames with high error as "bad" configurations

# Define success threshold (e.g., error < 10cm = success)
ERROR_THRESHOLD = 0.10  # meters

df['success'] = (df['error_m'] < ERROR_THRESHOLD).astype(int)

n_success = df['success'].sum()
n_failure = (df['success'] == 0).sum()
print(f"\nClass balance (using {ERROR_THRESHOLD*100:.0f}cm threshold):")
print(f"  Success (<{ERROR_THRESHOLD*100:.0f}cm): {n_success} ({n_success/len(df)*100:.1f}%)")
print(f"  Failure (≥{ERROR_THRESHOLD*100:.0f}cm): {n_failure} ({n_failure/len(df)*100:.1f}%)")

# ============================================================================
# SELECT FEATURES
# ============================================================================

# Per-match features (mean values per frame)
match_features = [
    "mean_ratio_test",
    "mean_n_candidates",
    "n_matches"
]

# Geometric features (NEW!)
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

# Combine all features
all_features = match_features + geometric_features

print(f"\nFeature set:")
print(f"  Match features:     {len(match_features)}")
print(f"  Geometric features: {len(geometric_features)}")
print(f"  Total:              {len(all_features)}")

# Check which features are available
available_features = [f for f in all_features if f in df.columns]
missing_features = [f for f in all_features if f not in df.columns]

if missing_features:
    print(f"\n⚠ Warning: Missing features: {missing_features}")
    all_features = available_features

print(f"\nUsing {len(all_features)} features:")
for feat in all_features:
    print(f"  - {feat}")

# Extract features and target
X = df[all_features].values
y = df["success"].values

# Handle NaN and inf values
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
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set:  {len(X_test)} samples")

# Build base pipeline
base_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
        C=1.0  # Can tune this
    ))
])

# Calibrate to get better probability estimates
print("\nTraining with calibration...")
pipeline = CalibratedClassifierCV(base_pipeline, method='isotonic', cv=5)
pipeline.fit(X_train, y_train)
print("✓ Training complete")

# ============================================================================
# EVALUATE
# ============================================================================

# Predict probabilities
probs_train = pipeline.predict_proba(X_train)[:, 1]
probs_test = pipeline.predict_proba(X_test)[:, 1]

# AUC
auc_train = roc_auc_score(y_train, probs_train)
auc_test = roc_auc_score(y_test, probs_test)

print(f"\n{'='*60}")
print(f"MODEL PERFORMANCE")
print(f"{'='*60}")
print(f"AUC (train): {auc_train:.3f}")
print(f"AUC (test):  {auc_test:.3f}")

# Classification report
preds = (probs_test > 0.5).astype(int)
print(f"\nClassification Report (test set):")
print(classification_report(y_test, preds, target_names=['Failure', 'Success']))

# Get base model coefficients
base_model = pipeline.calibrated_classifiers_[0].estimator
coef = base_model["clf"].coef_[0]
intercept = base_model["clf"].intercept_[0]

print(f"\n{'='*60}")
print(f"LEARNED WEIGHTS (Base Model)")
print(f"{'='*60}")
print(f"{'Feature':35s} {'Weight':>10s}")
print(f"{'-'*45}")

# Sort features by absolute weight
feature_importance = list(zip(all_features, coef))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

for name, weight in feature_importance:
    direction = "↓ confidence" if weight < 0 else "↑ confidence"
    marker = "***" if abs(weight) > 0.5 else ("**" if abs(weight) > 0.3 else "*")
    print(f"{name:35s} {weight:+10.4f}  {direction:15s} {marker}")

print(f"\nIntercept: {intercept:.4f}")

# Identify most important features
print(f"\n{'='*60}")
print(f"TOP 5 MOST IMPORTANT FEATURES")
print(f"{'='*60}")
for i, (name, weight) in enumerate(feature_importance[:5], 1):
    print(f"{i}. {name:30s} (weight: {weight:+.3f})")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================

print(f"\n{'='*60}")
print(f"CORRELATION WITH ACTUAL ERROR")
print(f"{'='*60}")

from scipy.stats import pearsonr

# We need to get the actual test set indices properly
# Since train_test_split was called with random_state=42, we can recreate the split
from sklearn.model_selection import train_test_split as tts

# Recreate the split just to get indices
_, test_idx_split, _, _ = tts(
    np.arange(len(df)), y, test_size=0.2, random_state=42, stratify=y
)

# Predicted uncertainty vs actual error
predicted_uncertainty = 1.0 / (probs_test + 0.01)  # Higher uncertainty for lower confidence
actual_errors = df.iloc[test_idx_split]['error_m'].values

if len(predicted_uncertainty) == len(actual_errors):
    r, p = pearsonr(predicted_uncertainty, actual_errors)
    print(f"\nCorrelation (predicted uncertainty vs actual error):")
    print(f"  Pearson r: {r:+.3f}")
    print(f"  p-value:   {p:.2e}")
    if p < 0.001:
        print(f"  ✓ Highly significant correlation!")
    elif p < 0.05:
        print(f"  ✓ Significant correlation")
    else:
        print(f"  ✗ Not significant (p > 0.05)")
else:
    print(f"⚠ Cannot compute correlation (dimension mismatch)")

# ============================================================================
# SAVE MODEL
# ============================================================================

os.makedirs('results', exist_ok=True)

model_data = {
    "model": pipeline,
    "features": all_features,
    "auc_train": auc_train,
    "auc_test": auc_test,
    "feature_importance": feature_importance
}

joblib.dump(model_data, "results/match_confidence_model.joblib")
print(f"\n✓ Saved model to results/match_confidence_model.joblib")

# ============================================================================
# VISUALIZE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confidence distributions
axes[0, 0].hist(probs_test[y_test == 0], bins=20, alpha=0.6, label='Failure', color='red')
axes[0, 0].hist(probs_test[y_test == 1], bins=20, alpha=0.6, label='Success', color='green')
axes[0, 0].set_xlabel('Predicted Confidence P(success)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Confidence Distribution (Test Set)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Calibration curve
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
actual_rates = []

for i in range(len(bins) - 1):
    mask = (probs_test >= bins[i]) & (probs_test < bins[i+1])
    count = mask.sum()
    if count > 0:
        actual_rates.append(y_test[mask].mean())
    else:
        actual_rates.append(np.nan)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
axes[0, 1].plot(bin_centers, actual_rates, 'o-', linewidth=2, markersize=8, label='Actual')
axes[0, 1].set_xlabel('Predicted Confidence')
axes[0, 1].set_ylabel('Actual Success Rate')
axes[0, 1].set_title('Calibration Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)

# Plot 3: Feature importances (absolute coefficients)
abs_coef = np.array([abs(w) for _, w in feature_importance])
feat_names = [n for n, _ in feature_importance]

# Take top 10
top_n = min(10, len(abs_coef))
axes[1, 0].barh(range(top_n), abs_coef[:top_n])
axes[1, 0].set_yticks(range(top_n))
axes[1, 0].set_yticklabels(feat_names[:top_n], fontsize=9)
axes[1, 0].set_xlabel('|Coefficient|')
axes[1, 0].set_title('Top 10 Feature Importances')
axes[1, 0].grid(axis='x', alpha=0.3)
axes[1, 0].invert_yaxis()

# Plot 4: Predicted uncertainty vs actual error (if available)
if len(predicted_uncertainty) == len(actual_errors):
    axes[1, 1].scatter(actual_errors * 100, predicted_uncertainty, alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Actual Error (cm)')
    axes[1, 1].set_ylabel('Predicted Uncertainty (a.u.)')
    axes[1, 1].set_title(f'Uncertainty vs Error (r={r:.3f}, p={p:.2e})')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].set_xlim(0, max(20, np.percentile(actual_errors * 100, 95)))
else:
    axes[1, 1].text(0.5, 0.5, 'Correlation plot\nunavailable', 
                    ha='center', va='center', fontsize=12)
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('results/match_confidence_geometric.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/match_confidence_geometric.png")

plt.show()

print(f"\n{'='*60}")
print("DONE - MODEL WITH GEOMETRIC FEATURES")
print(f"{'='*60}")