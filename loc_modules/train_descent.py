"""
Train match confidence model from lifecycle logs - Version 2
Now includes reprojection error for better confidence gradations.

Learns: P(match is correct | descriptor distance, ratio test, vocab stats, geometry)
Output: Logistic regression model saved to results/
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

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("="*60)
print("TRAINING MATCH CONFIDENCE MODEL v2")
print("="*60)

df = pd.read_csv("results/feature_lifecycles.csv")
print(f"\nLoaded {len(df)} feature lifecycles")

# Filter: only features that got matched (have descriptor distance)
df = df[df["best_match_distance"].notna()].copy()
print(f"Features that reached matching: {len(df)}")

# Check class balance
n_survivors = df['survived'].sum()
n_outliers = (~df['survived']).sum()
print(f"\n  Survivors (inliers): {n_survivors} ({n_survivors/len(df)*100:.1f}%)")
print(f"  Outliers:            {n_outliers} ({n_outliers/len(df)*100:.1f}%)")

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================

feature_cols = [
    "best_match_distance",      # Core signal
    "ratio_test_value",          # Lowe's ratio
    "n_candidates",              # Vocab ambiguity
    "reprojection_error",        # Geometric consistency (NEW!)
    "detector_score"             # XFeat confidence
]

# Extract features and target
X = df[feature_cols].values
y = df["survived"].astype(int).values

# Handle reprojection error properly:
# - Outliers have inf or very large values (they were rejected by RANSAC)
# - Replace inf/large values with a penalty value (e.g., 50 pixels)
# - This preserves the outliers in the dataset while giving them high error
reproj_idx = feature_cols.index("reprojection_error")
reproj_errors = X[:, reproj_idx].copy()

# Replace inf and extreme values with penalty
reproj_errors = np.nan_to_num(reproj_errors, nan=50.0, posinf=50.0, neginf=50.0)
reproj_errors = np.clip(reproj_errors, 0, 50.0)  # Clip to [0, 50] range
X[:, reproj_idx] = reproj_errors

print(f"\nReprojection error handling:")
print(f"  Outliers (should have high error): {(~df['survived']).sum()}")
print(f"  Mean error for outliers: {reproj_errors[y == 0].mean():.2f} px")
print(f"  Mean error for inliers:  {reproj_errors[y == 1].mean():.2f} px")

# Handle any remaining NaNs in other features
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
X = np.clip(X, -1e6, 1e6)  # Clip to reasonable range

print(f"Feature matrix: {X.shape}")
print(f"Target vector: {y.shape}")

# ============================================================================
# 3. TRAIN MODEL
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
        random_state=42
    ))
])

# Calibrate to get better probability estimates
print("\nTraining with calibration...")
pipeline = CalibratedClassifierCV(base_pipeline, method='isotonic', cv=5)
pipeline.fit(X_train, y_train)
print("✓ Training complete")

# ============================================================================
# 4. EVALUATE
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

# Classification report (at 0.5 threshold)
preds = (probs_test > 0.5).astype(int)
print(f"\nClassification Report (test set):")
print(classification_report(y_test, preds, target_names=['Outlier', 'Inlier']))

# Get base model coefficients (from first fold of calibrated model)
base_model = pipeline.calibrated_classifiers_[0].estimator
coef = base_model["clf"].coef_[0]
intercept = base_model["clf"].intercept_[0]

print(f"\n{'='*60}")
print(f"LEARNED WEIGHTS (Base Model)")
print(f"{'='*60}")
print(f"{'Feature':30s} {'Weight':>10s}")
print(f"{'-'*40}")
for name, weight in zip(feature_cols, coef):
    direction = "↑ confidence" if weight < 0 else "↓ confidence"
    print(f"{name:30s} {weight:+10.4f}  {direction}")
print(f"\nIntercept: {intercept:.4f}")

# ============================================================================
# 5. ANALYZE CONFIDENCE SPREAD
# ============================================================================

print(f"\n{'='*60}")
print(f"CONFIDENCE DISTRIBUTION ANALYSIS")
print(f"{'='*60}")

# For inliers only, show confidence spread
inlier_probs = probs_test[y_test == 1]
print(f"\nInlier confidence statistics:")
print(f"  Min:    {inlier_probs.min():.3f}")
print(f"  25th:   {np.percentile(inlier_probs, 25):.3f}")
print(f"  Median: {np.median(inlier_probs):.3f}")
print(f"  75th:   {np.percentile(inlier_probs, 75):.3f}")
print(f"  Max:    {inlier_probs.max():.3f}")

# Count how many in each confidence range
ranges = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 0.95), (0.95, 1.0)]
print(f"\nInlier confidence distribution:")
for low, high in ranges:
    count = ((inlier_probs >= low) & (inlier_probs < high)).sum()
    pct = count / len(inlier_probs) * 100
    print(f"  [{low:.2f}, {high:.2f}): {count:4d} ({pct:5.1f}%)")

# ============================================================================
# 6. SAVE MODEL
# ============================================================================

os.makedirs('results', exist_ok=True)

model_data = {
    "model": pipeline,
    "features": feature_cols,
    "auc_train": auc_train,
    "auc_test": auc_test
}

joblib.dump(model_data, "results/match_confidence_model.joblib")
print(f"\n✓ Saved model to results/match_confidence_model.joblib")

# ============================================================================
# 7. VISUALIZE
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Confidence distributions
axes[0, 0].hist(probs_test[y_test == 0], bins=30, alpha=0.6, label='Outliers', color='red')
axes[0, 0].hist(probs_test[y_test == 1], bins=30, alpha=0.6, label='Inliers', color='green')
axes[0, 0].set_xlabel('Predicted Confidence P(inlier)')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Confidence Distribution (Test Set)')
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: Calibration curve
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2
actual_rates = []
bin_counts = []

for i in range(len(bins) - 1):
    mask = (probs_test >= bins[i]) & (probs_test < bins[i+1])
    count = mask.sum()
    bin_counts.append(count)
    if count > 0:
        actual_rates.append(y_test[mask].mean())
    else:
        actual_rates.append(np.nan)

axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
axes[0, 1].plot(bin_centers, actual_rates, 'o-', linewidth=2, markersize=8, label='Actual')
axes[0, 1].set_xlabel('Predicted Confidence')
axes[0, 1].set_ylabel('Actual Inlier Rate')
axes[0, 1].set_title('Calibration Curve')
axes[0, 1].legend()
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xlim(0, 1)
axes[0, 1].set_ylim(0, 1)

# Plot 3: Feature importances (absolute coefficients)
abs_coef = np.abs(coef)
sorted_idx = np.argsort(abs_coef)
axes[1, 0].barh(range(len(feature_cols)), abs_coef[sorted_idx])
axes[1, 0].set_yticks(range(len(feature_cols)))
axes[1, 0].set_yticklabels([feature_cols[i] for i in sorted_idx])
axes[1, 0].set_xlabel('|Coefficient|')
axes[1, 0].set_title('Feature Importance')
axes[1, 0].grid(axis='x', alpha=0.3)

# Plot 4: Confidence vs Reprojection Error (for inliers only)
inlier_mask = y_test == 1
inlier_reproj = X_test[inlier_mask, reproj_idx]
inlier_conf = probs_test[inlier_mask]

axes[1, 1].scatter(inlier_reproj, inlier_conf, alpha=0.3, s=10)
axes[1, 1].set_xlabel('Reprojection Error (pixels)')
axes[1, 1].set_ylabel('Predicted Confidence')
axes[1, 1].set_title('Confidence vs Reprojection Error (Inliers)')
axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_xlim(0, 10)

plt.tight_layout()
plt.savefig('results/match_confidence_analysis_v2.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/match_confidence_analysis_v2.png")

plt.show()

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")