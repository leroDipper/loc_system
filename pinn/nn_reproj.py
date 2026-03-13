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

def parse_dist(s):
    vals = [float(x) for x in str(s).split(',')]
    arr = np.zeros(DIST_LEN, dtype=np.float32)
    arr[:min(len(vals), DIST_LEN)] = vals[:DIST_LEN]
    return arr

X_all = np.stack(df['reproj_error_dist'].apply(parse_dist).values)

# Train/test split — same as distillation NN
lab_df = df[df['env'] == 'lab'].sample(frac=1, random_state=42).reset_index(drop=True)
lab_train_idx = lab_df.index[:int(0.8 * len(lab_df))]
lab_test_idx = lab_df.index[int(0.8 * len(lab_df)):]

tum_fr1_df = df[df['env'] == 'tum_fr1'].sample(frac=1, random_state=42).reset_index(drop=True)
tum_fr1_train_idx = tum_fr1_df.index[:int(0.5 * len(tum_fr1_df))]
tum_fr1_test_idx = tum_fr1_df.index[int(0.5 * len(tum_fr1_df)):]

test_df = pd.concat([lab_df.iloc[int(0.8 * len(lab_df)):],
                     tum_fr1_df.iloc[int(0.5 * len(tum_fr1_df)):]],
                    ignore_index=True)
X_test = np.stack(test_df['reproj_error_dist'].apply(parse_dist).values)
y_test = test_df['combined_score'].values.astype(np.float32)

train_envs = ['mh_01', 'mh_03', 'mh_05', 'tum_fr2', 'tum_fr3']
balanced = []
for env in train_envs:
    env_df = df[df['env'] == env]
    balanced.append(env_df.sample(n=400, replace=len(env_df) < 400, random_state=42))
balanced.append(lab_df.iloc[:int(0.8 * len(lab_df))].sample(n=400, replace=True, random_state=42))
balanced.append(tum_fr1_df.iloc[:int(0.5 * len(tum_fr1_df))].sample(n=400, replace=True, random_state=42))
train_df = pd.concat(balanced, ignore_index=True)

X_train = np.stack(train_df['reproj_error_dist'].apply(parse_dist).values)
y_train = train_df['combined_score'].values.astype(np.float32)

# Normalise input
means = X_train.mean(axis=0)
stds = X_train.std(axis=0) + 1e-9
X_train = ((X_train - means) / stds).astype(np.float32)
X_test = ((X_test - means) / stds).astype(np.float32)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
                          batch_size=64, shuffle=True)

class ReprojDistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIST_LEN, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

model = ReprojDistNet()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()

print("Training model 3 — reprojection distribution...")
for epoch in range(500):
    model.train()
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimiser.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/500 - Loss: {epoch_loss:.4f}")

model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X_test)).numpy()

for env in ['tum_fr1', 'lab']:
    mask = test_df['env'] == env
    r = pearsonr(preds[mask], test_df['error_m'].values[mask]).statistic
    print(f"{env.upper()} (n={mask.sum()}): Pearson={r:+.3f}")

torch.save(model.state_dict(), 'results/reproj_dist_model.pth')
np.save('results/reproj_dist_norm.npy', np.stack([means, stds]))
print("Saved reproj_dist_model.pth and reproj_dist_norm.npy")