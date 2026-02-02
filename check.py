import pandas as pd
df = pd.read_csv('results/uncertainty_with_geometry.csv')
map_cols = ['mean_inlier_track_length', 'median_inlier_track_length', 
            'mean_inlier_ba_error', 'median_inlier_ba_error', 
            'frac_high_quality_inliers']
print(df[map_cols].corr())