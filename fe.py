import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression

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

# Remove outliers per environment
clean_parts = []
for env, group in df.groupby('env'):
    mean_err = group['error_m'].mean()
    std_err = group['error_m'].std()
    clean_parts.append(group[group['error_m'] < mean_err + 3 * std_err])
df_clean = pd.concat(clean_parts, ignore_index=True)
scores_clean = scores_01[df_clean.index]
errors_clean = df_clean['error_m'].values

# Bin-level monotonicity — 20 quantile bins, compute median per bin
df_temp = pd.DataFrame({'score': scores_clean, 'error_m': errors_clean})
df_temp['bin'] = pd.qcut(df_temp['score'], q=20, duplicates='drop')
bin_stats = df_temp.groupby('bin', observed=True)['error_m'].median().reset_index()
bin_stats['score_mid'] = bin_stats['bin'].apply(lambda x: (x.left + x.right) / 2)

# Keep only bins where median is monotonically decreasing with score
monotonic_bins = []
min_error = np.inf
for _, row in bin_stats.sort_values('score_mid', ascending=False).iterrows():
    if row['error_m'] <= min_error:
        min_error = row['error_m']
        monotonic_bins.append(row)

mono_df = pd.DataFrame(monotonic_bins).sort_values('score_mid')
print(f"Monotonic bins: {len(mono_df)} / {len(bin_stats)}")
print(f"\nScore mid → Median error:")
for _, row in mono_df.iterrows():
    print(f"  {row['score_mid']:.3f} → {row['error_m']*100:.1f} cm")

# Fit isotonic regression on bin medians
iso = IsotonicRegression(increasing=False, out_of_bounds='clip')
iso.fit(mono_df['score_mid'].values, mono_df['error_m'].values)

# Evaluate on all clean data
pred = iso.predict(scores_clean)
pearson = pearsonr(pred, errors_clean).statistic
mae = np.mean(np.abs(pred - errors_clean))

print(f"\nPearson: {pearson:+.3f}")
print(f"MAE: {mae*100:.2f} cm")

print(f"\nExample scores:")
for s in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]:
    u = iso.predict([s])[0]
    print(f"  score={s:.2f} → {u*100:.1f} cm")

joblib.dump(iso, 'results/uncertainty_calibration.joblib')
print(f"\nSaved to results/uncertainty_calibration.joblib")