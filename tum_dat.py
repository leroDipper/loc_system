import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Load all sequences
dfs = []
for name, path in [
    ('mh_01', 'results/mh_01_uncertainty250.csv'),
    ('mh_03', 'results/mh_03_uncertainty250.csv'),
    ('mh_05', 'results/mh_05_uncertainty250.csv'),
    ('tum_fr1', 'results/tum_fr1_uncertainty250.csv'),
    ('tum_fr2', 'results/tum_fr2_uncertainty250.csv'),
    ('tum_fr3', 'results/tum_fr3_uncertainty250.csv'),
]:
    df = pd.read_csv(path)
    df['env'] = name
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True).dropna()

# Target
df = df[df['translation_std_total_m'] > 0]
df['log_k'] = np.log(df['error_m'].clip(1e-4) / df['translation_std_total_m'].clip(1e-6))

features = ['mean_inverse_depth', 'img_brightness', 'img_blur_score', 'depth_mean', 'depth_range', 'pnp_condition_number']

X = df[features].values
y = df['log_k'].values

pipe = make_pipeline(StandardScaler(), Ridge())
r2 = cross_val_score(pipe, X, y, cv=5, scoring='r2').mean()
corr = np.corrcoef(pipe.fit(X, y).predict(X), y)[0,1]
print(f"log(k) linear predictability: R2={r2:.3f}, correlation={corr:.3f}")
print(f"log(k) stats: mean={y.mean():.3f}, std={y.std():.3f}")


from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

features = ['mean_inverse_depth', 'img_brightness', 'img_blur_score', 'depth_mean', 'depth_range', 'pnp_condition_number']

X = df[features].values
y = df['log_k'].values

pipe = make_pipeline(StandardScaler(), Ridge())
pipe.fit(X, y)

df['k_pred'] = np.exp(pipe.predict(X))
df['uncertainty_corrected'] = df['k_pred'] * df['translation_std_total_m']

for env in df['env'].unique():
    sub = df[df['env'] == env]
    corr_raw = np.corrcoef(sub['error_m'], sub['translation_std_total_m'])[0,1]
    corr_corrected = np.corrcoef(sub['error_m'], sub['uncertainty_corrected'])[0,1]
    print(f"{env}: physics={corr_raw:.3f} → corrected={corr_corrected:.3f}")


for test_env in df['env'].unique():
    train = df[df['env'] != test_env]
    test = df[df['env'] == test_env]
    
    pipe = make_pipeline(StandardScaler(), Ridge())
    pipe.fit(train[features].values, train['log_k'].values)
    
    k_pred = np.exp(pipe.predict(test[features].values))
    uncertainty_corrected = k_pred * test['translation_std_total_m']
    
    corr_raw = np.corrcoef(test['error_m'], test['translation_std_total_m'])[0,1]
    corr_corrected = np.corrcoef(test['error_m'], uncertainty_corrected)[0,1]
    print(f"{test_env}: physics={corr_raw:.3f} → corrected={corr_corrected:.3f}")


import matplotlib.pyplot as plt

mh03 = df[df['env'] == 'mh_03']
plt.scatter(mh03['mean_inverse_depth'], np.log(mh03['error_m'].clip(1e-4) / mh03['translation_std_total_m'].clip(1e-6)), alpha=0.3)
plt.xlabel('mean_inverse_depth')
plt.ylabel('log(k)')
plt.title('MH_03: log(k) vs mean_inverse_depth')
plt.savefig('mh03_logk.png')


mh03 = df[df['env'] == 'mh_03']
corr = np.corrcoef(mh03['pnp_condition_number'], np.log(mh03['error_m'].clip(1e-4) / mh03['translation_std_total_m'].clip(1e-6)))[0,1]
print(f"condition_estimate vs log(k) on MH_03: {corr:.3f}")