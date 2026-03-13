import pandas as pd
import numpy as np
import joblib

df = pd.read_csv('results/training_data.csv')

# Remove outliers per environment
clean_parts = []
for env, group in df.groupby('env'):
    mean_err = group['error_m'].mean()
    std_err = group['error_m'].std()
    clean_parts.append(group[group['error_m'] < mean_err + 3 * std_err])
df_clean = pd.concat(clean_parts, ignore_index=True)

# Save the sorted error array — this is the empirical CDF
errors_sorted = np.sort(df_clean['error_m'].values)
joblib.dump(errors_sorted, 'results/empirical_cdf.joblib')

print(f"CDF built from {len(errors_sorted)} frames")
print(f"\nExample lookups:")
for s in [0.2, 0.4, 0.6, 0.8, 0.9, 0.95]:
    q = (1 - s) * 100
    est = np.percentile(errors_sorted, q)
    low = np.percentile(errors_sorted, max(0, q - 10))
    high = np.percentile(errors_sorted, min(100, q + 10))
    print(f"  score={s:.2f} → {est*100:.1f}cm [{low*100:.1f} - {high*100:.1f}cm]")