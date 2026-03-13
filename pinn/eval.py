import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr

torch.manual_seed(42)
np.random.seed(42)

df = pd.read_csv('results/training_data.csv')

FEATURES = ['physics', 'projection', 'pose_jump',
            'mean_alignment_error', 'anisotropy',
            'principal_axis_x', 'principal_axis_y', 'principal_axis_z',
            'condition_3d', 'trans_anisotropy', 'trans_max_std']

ENVIRONMENTS = ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3', 'lab']

class UncertaintyNet(nn.Module):
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

results = {}

for held_out in ENVIRONMENTS:
    torch.manual_seed(42)
    np.random.seed(42)

    test_df = df[df['env'] == held_out].reset_index(drop=True)
    train_envs = [e for e in ENVIRONMENTS if e != held_out]

    balanced = []
    for env in train_envs:
        env_df = df[df['env'] == env]
        balanced.append(env_df.sample(n=400, replace=len(env_df) < 400, random_state=42))
    train_df = pd.concat(balanced, ignore_index=True)

    means = train_df[FEATURES].mean()
    stds = train_df[FEATURES].std() + 1e-9

    def prepare(data):
        return ((data[FEATURES] - means) / stds).values.astype(np.float32)

    X_train = prepare(train_df)
    y_train = train_df['combined_score'].values.astype(np.float32)
    X_test = prepare(test_df)

    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                        batch_size=64, shuffle=True)

    model = UncertaintyNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.L1Loss()

    model.train()
    for epoch in range(500):
        for X_batch, y_batch in loader:
            opt.zero_grad()
            criterion(model(X_batch), y_batch).backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy()

    r = pearsonr(preds, test_df['error_m'].values).statistic
    results[held_out] = r
    print(f"  {held_out.upper():10s} (n={len(test_df):4d}): Pearson={r:+.3f}")

print(f"\nMean Pearson: {np.mean(list(results.values())):+.3f}")