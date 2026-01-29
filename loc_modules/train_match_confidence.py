"""
Train match confidence model from lifecycle logs.

Learns: P(match is correct | descriptor distance, ratio test, vocab stats)
Output: Logistic regression model saved to results/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import matplotlib.pyplot as plt
import os

# ============================================================================
# 1. LOAD DATA
# ============================================================================

print("="*60)
print("TRAINING MATCH CONFIDENCE MODEL")
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
    "detector_score"             # XFeat confidence
]

# Extract features and target
X = df[feature_cols].values
y = df["survived"].astype(int).values

# Handle any NaNs and clip extreme values to prevent overflow
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
X = np.clip(X, -1e6, 1e6)  # Clip to reasonable range

print(f"\nFeature matrix: {X.shape}")
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

# Build pipeline with scaling + logistic regression
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        C=0.1,
        max_iter=1000,
        class_weight="balanced",  # Handle class imbalance
        random_state=42
    ))
])

# Train
print("\nTraining...")
pipeline.fit(X_train, y_train)
print("✓ Training complete")

from sklearn.calibration import CalibratedClassifierCV

calibrated_model = CalibratedClassifierCV(pipeline, method='isotonic', cv=5)
calibrated_model.fit(X_train, y_train)

# ============================================================================
# 4. EVALUATE
# ============================================================================

probs_train = calibrated_model.predict_proba(X_train)[:, 1]
probs_test = calibrated_model.predict_proba(X_test)[:, 1]



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

# Learned coefficients
coef = pipeline["clf"].coef_[0]
intercept = pipeline["clf"].intercept_[0]

print(f"\n{'='*60}")
print(f"LEARNED WEIGHTS")
print(f"{'='*60}")
print(f"{'Feature':30s} {'Weight':>10s}")
print(f"{'-'*40}")
for name, weight in zip(feature_cols, coef):
    direction = "↑ confidence" if weight < 0 else "↓ confidence"
    print(f"{name:30s} {weight:+10.4f}  {direction}")
print(f"\nIntercept: {intercept:.4f}")

# ============================================================================
# 5. SAVE MODEL
# ============================================================================

os.makedirs('results', exist_ok=True)

model_data = {
    "model": calibrated_model,
    "features": feature_cols,
    "auc_train": auc_train,
    "auc_test": auc_test
}

joblib.dump(model_data, "results/match_confidence_model.joblib")
print(f"\n✓ Saved model to results/match_confidence_model.joblib")

# ============================================================================
# 6. VISUALIZE
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Confidence distributions
axes[0].hist(probs_test[y_test == 0], bins=30, alpha=0.6, label='Outliers', color='red')
axes[0].hist(probs_test[y_test == 1], bins=30, alpha=0.6, label='Inliers', color='green')
axes[0].set_xlabel('Predicted Confidence P(inlier)')
axes[0].set_ylabel('Count')
axes[0].set_title('Confidence Distribution (Test Set)')
axes[0].legend()
axes[0].grid(alpha=0.3)

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

axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
axes[1].plot(bin_centers, actual_rates, 'o-', linewidth=2, markersize=8, label='Actual')
axes[1].set_xlabel('Predicted Confidence')
axes[1].set_ylabel('Actual Inlier Rate')
axes[1].set_title('Calibration Curve')
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

# Plot 3: Feature importances (absolute coefficients)
abs_coef = np.abs(coef)
sorted_idx = np.argsort(abs_coef)
axes[2].barh(range(len(feature_cols)), abs_coef[sorted_idx])
axes[2].set_yticks(range(len(feature_cols)))
axes[2].set_yticklabels([feature_cols[i] for i in sorted_idx])
axes[2].set_xlabel('|Coefficient|')
axes[2].set_title('Feature Importance')
axes[2].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('results/match_confidence_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to results/match_confidence_analysis.png")

plt.show()

print(f"\n{'='*60}")
print("DONE")
print(f"{'='*60}")