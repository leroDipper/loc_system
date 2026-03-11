"""
Evaluation script for trained MAML uncertainty model.
Evaluates on MH_01 (in-distribution) and lab (out-of-distribution).
"""

import torch
import numpy as np
import pandas as pd
from net import UncertaintyNet
from dataObjects import LocalisationDataset
from train import evaluate, compute_metrics
from dataLoader import apply_transforms


# ============================================================================
# LOAD MODEL
# ============================================================================

checkpoint = torch.load('results/maml_uncertainty_model.pt', weights_only=False)
scaler = checkpoint['scaler']
features = checkpoint['features']
n_features = checkpoint['n_features']

model = UncertaintyNet(n_features)
model.load_state_dict(checkpoint['model_state'])
model.eval()

print(f"Loaded model with {n_features} features")


# ============================================================================
# HELPER
# ============================================================================

def build_env(name, csv_path, scaler, features):
    df = pd.read_csv(csv_path)
    df = apply_transforms(df)
    df = df[df['error_m'] < df['error_m'].mean() + 3 * df['error_m'].std()].copy()

    X = scaler.transform(df[features].values)
    X = np.clip(X, -5, 5)
    y = df['error_m'].values
    threshold = np.median(y) * 1.5

    env = LocalisationDataset(name, X, y, threshold)
    print(f"  {name}: {len(df)} frames, mean={y.mean()*100:.1f}cm, "
          f"median={np.median(y)*100:.1f}cm, threshold={threshold*100:.1f}cm")
    return env


# ============================================================================
# EVALUATE
# ============================================================================

print("\n" + "="*60)
print("IN-DISTRIBUTION EVALUATION (EuRoC)")
print("="*60)

for seq in ['mh_01', 'mh_03', 'mh_05']:
    print(f"\n{seq.upper()}:")
    env = build_env(seq, f'results/{seq}_uncertainty250.csv', scaler, features)
    predictions, uncertainties, p_failure, actuals = evaluate(model, env)

    print(f"  Prediction diagnostics:")
    print(f"    Mean predicted: {predictions.mean()*100:.2f}cm")
    print(f"    Std predicted:  {predictions.std()*100:.2f}cm")
    print(f"    Mean actual:    {actuals.mean()*100:.2f}cm")
    print(f"    Std actual:     {actuals.std()*100:.2f}cm")
    print(f"    Mean uncertainty: {uncertainties.mean()*100:.2f}cm")

    compute_metrics(predictions, uncertainties, p_failure, actuals, env.failure_threshold)


print("\n" + "="*60)
print("OUT-OF-DISTRIBUTION EVALUATION (Lab)")
print("="*60)

print("\nLAB:")
env = build_env('lab', 'results/lab_uncertainty250.csv', scaler, features)
predictions, uncertainties, p_failure, actuals = evaluate(model, env)

print(f"  Prediction diagnostics:")
print(f"    Mean predicted: {predictions.mean()*100:.2f}cm")
print(f"    Std predicted:  {predictions.std()*100:.2f}cm")
print(f"    Mean actual:    {actuals.mean()*100:.2f}cm")
print(f"    Std actual:     {actuals.std()*100:.2f}cm")
print(f"    Mean uncertainty: {uncertainties.mean()*100:.2f}cm")

compute_metrics(predictions, uncertainties, p_failure, actuals, env.failure_threshold)