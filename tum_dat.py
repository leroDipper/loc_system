import pandas as pd

df = pd.read_csv('results/mh_01_uncertainty250.csv')
print(df['frame'].head(20).tolist())