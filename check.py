import pandas as pd
import numpy as np

df = pd.read_csv('results/tum_fr1_uncertainty250.csv')
corr = np.corrcoef(df['error_m'], df['translation_std_total_m'])[0,1]
print(f"correlation(error_m, translation_std_total_m) = {corr:.3f}")
print(f"mean translation_std: {df['translation_std_total_m'].mean()*100:.2f}cm")
print(f"mean error:           {df['error_m'].mean()*100:.2f}cm")

import pandas as pd
import numpy as np

df = pd.read_csv('results/tum_fr1_uncertainty250.csv')
print(f"mean inlier_ba_error: {df['mean_inlier_ba_error'].mean():.3f}")
print(f"mean inlier_track_length: {df['mean_inlier_track_length'].mean():.3f}")

# What sigma_point_sq looks like
sigma_sq = (df['mean_inlier_ba_error']**2) / df['mean_inlier_track_length']
print(f"mean sigma_point_sq: {sigma_sq.mean():.6f}")
print(f"sigma2 from reproj: {(df['mean_inlier_reproj_error']**2).mean():.6f}")