import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import joblib

torch.manual_seed(42)
np.random.seed(42)

DIST_LEN = 50

df = pd.read_csv('results/training_data.csv')

# ── Feature definitions ──────────────────────────────────────────────────────

PHYSICS_FEATURES = ['physics', 'projection', 'pose_jump',
                    'mean_alignment_error', 'anisotropy',
                    'principal_axis_x', 'principal_axis_y', 'principal_axis_z']

MATCH_FEATURES = ['mean_inlier_reproj_error', 'std_inlier_reproj_error',
                  'match_spread_normalized', 'depth_relative_std', 'condition_estimate',
                  'mean_inlier_track_length', 'mean_inlier_ba_error',
                  'frac_high_quality_inliers', 'n_matches', 'n_inliers']

def parse_dist(s):
    vals = [float(x) for x in str(s).split(',')]
    arr = np.zeros(DIST_LEN, dtype=np.float32)
    arr[:min(len(vals), DIST_LEN)] = vals[:DIST_LEN]
    return arr

# ── Load all three models ────────────────────────────────────────────────────

class PhysicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class MatchQualityNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

class ReprojDistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DIST_LEN, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

physics_model = PhysicsNet()
physics_model.load_state_dict(torch.load('results/uncertainty_nn.pth', weights_only=False))
physics_model.eval()

match_model = MatchQualityNet()
match_model.load_state_dict(torch.load('results/match_quality_model.pth', weights_only=False))
match_model.eval()

reproj_model = ReprojDistNet()
reproj_model.load_state_dict(torch.load('results/reproj_dist_model.pth', weights_only=False))
reproj_model.eval()

# ── Load normalisation stats ─────────────────────────────────────────────────


physics_stats = pd.read_csv('results/feature_stats.csv', index_col=0)
physics_means = physics_stats['mean']
physics_stds = physics_stats['std']


reproj_norm = np.load('results/reproj_dist_norm.npy')
reproj_means = reproj_norm[0]
reproj_stds = reproj_norm[1]

# ── Get predictions from all three models on full dataset ────────────────────

def get_physics_preds(data):
    X = ((data[PHYSICS_FEATURES] - physics_means[PHYSICS_FEATURES].values) /
         physics_stds[PHYSICS_FEATURES].values).values.astype(np.float32)
    with torch.no_grad():
        return physics_model(torch.tensor(X)).numpy()

def get_match_preds(data, means, stds):
    X = ((data[MATCH_FEATURES] - means) / stds).values.astype(np.float32)
    with torch.no_grad():
        return match_model(torch.tensor(X)).numpy()

def get_reproj_preds(data):
    X = np.stack(data['reproj_error_dist'].apply(parse_dist).values)
    X = ((X - reproj_means) / reproj_stds).astype(np.float32)
    with torch.no_grad():
        return reproj_model(torch.tensor(X)).numpy()

# ── Train/test split ─────────────────────────────────────────────────────────

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

# Fit match normalisation on training data
match_means = train_df[MATCH_FEATURES].mean()
match_stds = train_df[MATCH_FEATURES].std() + 1e-9

# Get predictions
p1_train = get_physics_preds(train_df)
p2_train = get_match_preds(train_df, match_means, match_stds)
p3_train = get_reproj_preds(train_df)

p1_test = get_physics_preds(test_df)
p2_test = get_match_preds(test_df, match_means, match_stds)
p3_test = get_reproj_preds(test_df)

X_train = np.stack([p1_train, p2_train, p3_train], axis=1).astype(np.float32)
y_train = train_df['error_m'].values.astype(np.float32)

X_test = np.stack([p1_test, p2_test, p3_test], axis=1).astype(np.float32)
y_test = test_df['error_m'].values.astype(np.float32)

# ── Meta-learner ─────────────────────────────────────────────────────────────

class MetaLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

meta = MetaLearner()
optimiser = torch.optim.Adam(meta.parameters(), lr=1e-3)
criterion = nn.L1Loss()

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=64, shuffle=True)

print("Training meta-learner...")
for epoch in range(500):
    meta.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        loss = criterion(meta(X_batch), y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/500 - Loss: {epoch_loss:.4f}")

meta.eval()
with torch.no_grad():
    preds = meta(torch.tensor(X_test)).numpy()

print("\nResults:")
for env in ['tum_fr1', 'lab']:
    mask = test_df['env'] == env
    r = pearsonr(preds[mask], test_df['error_m'].values[mask]).statistic
    print(f"  {env.upper()} (n={mask.sum()}): Pearson={r:+.3f}")

torch.save(meta.state_dict(), 'results/meta_learner.pth')
joblib.dump({'match_means': match_means, 'match_stds': match_stds}, 'results/match_norm.joblib')
print("\nSaved meta_learner.pth and match_norm.joblib")