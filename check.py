import pandas as pd
import numpy as np

for name in ['mh_01', 'mh_03', 'mh_05', 'tum_fr1']:
    df = pd.read_csv(f'results/{name}_uncertainty250.csv')
    if 'translation_std_total_m' in df.columns:
        corr = np.corrcoef(df['error_m'], df['translation_std_total_m'])[0,1]
        print(f"{name}: correlation = {corr:.3f}")
    else:
        print(f"{name}: translation_std_total_m not in columns")
        print(f"  columns: {list(df.columns)}")