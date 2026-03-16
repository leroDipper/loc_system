import pandas as pd
import numpy as np

df = pd.read_csv('results/training_data.csv')
for env in df['env'].unique():
    e = df[df['env'] == env]
    r = np.corrcoef(e['projection'], e['error_m'])[0,1]
    print(f"{env}: projection corr={r:+.3f}  mean_proj={e['projection'].mean():.4f}")