import pandas as pd
import numpy as np

df = pd.read_csv('results/training_data.csv')

for env in ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3', 'lab']:
    e = df[df['env'] == env].copy()
    e['condition_3d'] = e['condition_3d'].clip(upper=e['condition_3d'].quantile(0.99))
    r = np.corrcoef(e['condition_3d'], e['error_m'])[0,1]
    print(f"{env}: condition_3d corr={r:+.3f}  (99th={e['condition_3d'].quantile(0.99):.1f})")