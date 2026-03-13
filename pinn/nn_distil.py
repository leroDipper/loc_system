import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import joblib

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv('results/training_data.csv')

features = ['physics', 'projection', 'pose_jump',
            'mean_alignment_error', 'anisotropy',
            'principal_axis_x', 'principal_axis_y', 'principal_axis_z', 'condition_3d','trans_max_std','trans_anisotropy']

# Train/test split
lab_df = df[df['env'] == 'lab'].sample(frac=1, random_state=42).reset_index(drop=True)
lab_train = lab_df.iloc[:int(0.8 * len(lab_df))]
lab_test = lab_df.iloc[int(0.8 * len(lab_df)):]

tum_fr1_df = df[df['env'] == 'tum_fr1'].sample(frac=1, random_state=42).reset_index(drop=True)
tum_fr1_train = tum_fr1_df.iloc[:int(0.5 * len(tum_fr1_df))]
tum_fr1_test = tum_fr1_df.iloc[int(0.5 * len(tum_fr1_df)):]

test_df = pd.concat([tum_fr1_test, lab_test], ignore_index=True)

train_envs = ['mh_01', 'mh_03', 'mh_05', 'tum_fr2', 'tum_fr3']
balanced = []
for env in train_envs:
    env_df = df[df['env'] == env]
    balanced.append(env_df.sample(n=400, replace=len(env_df) < 400, random_state=42))
balanced.append(lab_train.sample(n=400, replace=True, random_state=42))
balanced.append(tum_fr1_train.sample(n=400, replace=True, random_state=42))
train_df = pd.concat(balanced, ignore_index=True)

means = train_df[features].mean()
stds = train_df[features].std() + 1e-9

def prepare_X(data):
    return ((data[features] - means) / stds).values.astype(np.float32)

X_train = prepare_X(train_df)
y_train = train_df['combined_score'].values.astype(np.float32)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class UncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = UncertaintyNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

print("Training distillation NN...")
for epoch in range(500):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch + 1) % 100 == 0:
        print(f"  Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}")

# Get NN scores on all training data to fit normalization
model.eval()
X_all = prepare_X(train_df)
with torch.no_grad():
    all_scores = model(torch.tensor(X_all)).numpy()

# Fit min/max normalization from training distribution
score_min = float(np.percentile(all_scores, 1))
score_max = float(np.percentile(all_scores, 99))

def to_01(scores):
    clipped = np.clip(scores, score_min, score_max)
    return (clipped - score_min) / (score_max - score_min + 1e-9)

# Evaluate on test set
X_test = prepare_X(test_df)
with torch.no_grad():
    test_scores = model(torch.tensor(X_test)).numpy()

test_scores_01 = to_01(test_scores)
true_error = test_df['error_m'].values

print(f"\nTest set overall (n={len(test_df)})")
print(f"  Pearson(score, error_m): {pearsonr(test_scores_01, true_error).statistic:+.3f}")

for env in test_df['env'].unique():
    mask = test_df['env'].values == env
    r = pearsonr(test_scores_01[mask], true_error[mask]).statistic
    print(f"\n  {env.upper()} (n={mask.sum()})")
    print(f"    Pearson: {r:+.3f}")
    print(f"    Score range: [{test_scores_01[mask].min():.3f}, {test_scores_01[mask].max():.3f}]")
    print(f"    Mean score:  {test_scores_01[mask].mean():.3f}")

# Save
torch.save(model.state_dict(), 'results/uncertainty_nn.pth')
pd.DataFrame({'mean': means, 'std': stds}).to_csv('results/feature_stats.csv')
norm_stats = {'score_min': score_min, 'score_max': score_max}
joblib.dump(norm_stats, 'results/score_normalization.joblib')
print("\nSaved: uncertainty_nn.pth, feature_stats.csv, score_normalization.joblib")