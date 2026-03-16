import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, rankdata
import joblib

torch.manual_seed(42)
np.random.seed(42)

# ── Config ───────────────────────────────────────────────────────────────────

LAMBDA_PHYSICS = 0.05
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-3

FEATURES = ['physics', 'projection', 'pose_jump',
            'mean_alignment_error', 'anisotropy',
            'principal_axis_x', 'principal_axis_y', 'principal_axis_z',
            'condition_3d', 'trans_anisotropy', 'trans_max_std']

# ── Data ─────────────────────────────────────────────────────────────────────

df = pd.read_csv('results/training_data.csv')

lab_df = df[df['env'] == 'lab'].sample(frac=1, random_state=42).reset_index(drop=True)
tum_fr1_df = df[df['env'] == 'tum_fr1'].sample(frac=1, random_state=42).reset_index(drop=True)

test_df = pd.concat([lab_df.iloc[int(0.8 * len(lab_df)):],
                     tum_fr1_df.iloc[int(0.5 * len(tum_fr1_df)):]],
                    ignore_index=True)

train_envs = ['mh_01', 'mh_03', 'mh_05', 'tum_fr2', 'tum_fr3']
balanced = []
for env in train_envs:
    env_df = df[df['env'] == env]
    balanced.append(env_df.sample(n=400, replace=len(env_df) < 400, random_state=42))
balanced.append(lab_df.iloc[:int(0.8 * len(lab_df))].sample(n=400, replace=True, random_state=42))
balanced.append(tum_fr1_df.iloc[:int(0.5 * len(tum_fr1_df))].sample(n=400, replace=True, random_state=42))
train_df = pd.concat(balanced, ignore_index=True)

means = train_df[FEATURES].mean()
stds = train_df[FEATURES].std() + 1e-9

def prepare_X(data):
    return ((data[FEATURES] - means) / stds).values.astype(np.float32)

X_train = prepare_X(train_df)
y_train = train_df['combined_score'].values.astype(np.float32)
crb_raw = train_df['translation_std_crb'].values
crb_train = (rankdata(crb_raw) / len(crb_raw)).astype(np.float32)
X_test = prepare_X(test_df)

# Save normalisation stats
stats_df = pd.DataFrame({'feature': FEATURES, 'mean': means.values, 'std': stds.values})
stats_df.to_csv('results/feature_stats.csv', index=False)

# ── Model ────────────────────────────────────────────────────────────────────

class PINNUncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES), 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = PINNUncertaintyNet()
optimiser = torch.optim.Adam(model.parameters(), lr=LR)

# ── Loss ─────────────────────────────────────────────────────────────────────

def pinn_loss(predictions, targets, crb_rank):
    l_mae = torch.mean(torch.abs(predictions - targets))
    l_crb = torch.mean(torch.relu(crb_rank - predictions))
    return l_mae + LAMBDA_PHYSICS * l_crb

# ── Training ─────────────────────────────────────────────────────────────────

dataset = TensorDataset(torch.tensor(X_train),
                        torch.tensor(y_train),
                        torch.tensor(crb_train))
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Training PINN uncertainty estimator...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch, crb_batch in loader:
        optimiser.zero_grad()
        loss = pinn_loss(model(X_batch), y_batch, crb_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(loader.dataset)
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}: loss={epoch_loss:.4f}")

# ── Evaluation ───────────────────────────────────────────────────────────────

model.eval()
with torch.no_grad():
    raw_scores = model(torch.tensor(X_test)).numpy()

# Invert: high score = certain (low uncertainty), low score = uncertain
p1, p99 = np.percentile(raw_scores, 1), np.percentile(raw_scores, 99)
scores = 1.0 - np.clip((raw_scores - p1) / (p99 - p1 + 1e-9), 0, 1)

print(f"\nTest set overall (n={len(test_df)})")
# High score = low error, so correlation should be negative — report abs
r_overall = pearsonr(scores, test_df['error_m'].values).statistic
print(f"  Pearson(score, error_m): {r_overall:+.3f}  (abs={abs(r_overall):.3f})")

for env in ['tum_fr1', 'lab']:
    mask = test_df['env'] == env
    r = pearsonr(scores[mask], test_df['error_m'].values[mask]).statistic
    print(f"  {env.upper()} (n={mask.sum()})")
    print(f"    Pearson: {r:+.3f}  (abs={abs(r):.3f})")
    print(f"    Score range: [{scores[mask].min():.3f}, {scores[mask].max():.3f}]")
    print(f"    Mean score:  {scores[mask].mean():.3f}")

# ── Save model and CDF data ───────────────────────────────────────────────────

torch.save(model.state_dict(), 'results/uncertainty_nn.pth')
joblib.dump({'score_min': float(p1), 'score_max': float(p99)},
            'results/score_normalisation.joblib')

# Save training errors for empirical CDF
# Use all training data — outlier cleaned at 99th percentile per environment
cdf_errors = []
for env in df['env'].unique():
    env_errors = df[df['env'] == env]['error_m'].values
    p99_env = np.percentile(env_errors, 99)
    cdf_errors.extend(env_errors[env_errors <= p99_env].tolist())

cdf_errors = np.sort(cdf_errors)
joblib.dump(cdf_errors, 'results/empirical_cdf.joblib')

print(f"\nSaved: uncertainty_nn.pth, feature_stats.csv, score_normalisation.joblib")
print(f"Saved: empirical_cdf.joblib ({len(cdf_errors)} frames)")