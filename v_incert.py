import pandas as pd
import numpy as np

df = pd.read_csv('results/mh_01_uncertainty250.csv').dropna()

k = df['error_m'] / df['translation_std_total_m']
print(f"Correction factor k statistics:")
print(f"  mean:   {k.mean():.3f}")
print(f"  median: {k.median():.3f}")
print(f"  std:    {k.std():.3f}")
print(f"  min:    {k.min():.3f}")
print(f"  max:    {k.max():.3f}")

df['k'] = df['error_m'] / df['translation_std_total_m']

df['log_mean_inlier_track_length']   = np.log1p(df['mean_inlier_track_length'])
df['log_mean_inverse_depth']         = np.log1p(df['mean_inverse_depth'])
df['log_median_inlier_reproj_error'] = np.log1p(df['median_inlier_reproj_error'])
df['log_img_blur_score']             = np.log1p(df['img_blur_score'])

df['k'] = df['error_m'] / df['translation_std_total_m']

print("Correlation with correction factor k:")
correlations = {}
for col in df.columns:
    if col in ['error_m', 'k', 'translation_std_total_m', 'frame']:
        continue
    try:
        corr = np.corrcoef(df[col], df['k'])[0,1]
        correlations[col] = corr
    except:
        continue

for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {feat}: {corr:+.3f}")