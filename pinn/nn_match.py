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

features = ['mean_inlier_reproj_error', 'std_inlier_reproj_error', 
                     'match_spread_normalized', 'depth_relative_std', 'condition_estimate', 
                     'mean_inlier_track_length', 'mean_inlier_ba_error', 'frac_high_quality_inliers', 'n_matches', 'n_inliers']

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

class MatchQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)
    
model = MatchQualityNet()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

# Training loop
print("Starting training...")
epochs = 500
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), 'results/match_quality_model.pth')
print("Model saved as 'match_quality_model.pth'")

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

