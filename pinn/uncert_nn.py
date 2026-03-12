import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

df = pd.read_csv('results/training_data.csv')
feature_stats = pd.read_csv('results/feature_stats.csv', index_col=0)
norm_stats = joblib.load('results/score_normalization.joblib')

features = ['physics', 'projection', 'pose_jump',
            'mean_alignment_error', 'anisotropy',
            'principal_axis_x', 'principal_axis_y', 'principal_axis_z']

means = feature_stats['mean']
stds = feature_stats['std']

class UncertaintyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = UncertaintyNet()
model.load_state_dict(torch.load('results/uncertainty_nn.pth'))
model.eval()

X = ((df[features] - means) / stds).values.astype('float32')
with torch.no_grad():
    scores = model(torch.tensor(X)).numpy()

score_min = norm_stats['score_min']
score_max = norm_stats['score_max']
scores_01 = 1 - np.clip((scores - score_min) / (score_max - score_min + 1e-9), 0, 1)

df['score'] = scores_01

# Show error distribution per score bin
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
df['bin'] = pd.cut(df['score'], bins=bins)
print(df.groupby('bin')['error_m'].describe()[['count', 'mean', 'std', 'min', 'max']] * 100)