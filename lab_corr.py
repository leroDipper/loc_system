import pandas as pd
import numpy as np

def normalise(s):
    return (s - s.mean()) / (s.std() + 1e-9)

df = pd.read_csv('results/lab_uncertainty250.csv')
df = df.sort_values('frame').reset_index(drop=True)

# Pose jump velocity - lab uses sequential frame names not timestamps
# Try extracting number from filename
df['frame_num'] = df['frame'].str.replace('.jpg', '').astype(float)
positions = df[['C_x', 'C_y', 'C_z']].values
jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
dt = np.diff(df['frame_num'].values)
velocity = jumps / (dt + 1e-9)
df['pose_jump_velocity'] = np.concatenate([[velocity.mean()], velocity])

corr_phys = np.corrcoef(df['translation_std_total_m'], df['error_m'])[0, 1]
corr_jump = np.corrcoef(df['pose_jump_velocity'], df['error_m'])[0, 1]

w_phys = max(corr_phys, 0)
w_jump = max(corr_jump, 0)
total = w_phys + w_jump
w_phys /= total
w_jump /= total

combined = w_phys * normalise(df['translation_std_total_m']) + w_jump * normalise(df['pose_jump_velocity'])
corr_combined = np.corrcoef(combined, df['error_m'])[0, 1]

print(f"LAB (n={len(df)})")
print(f"  Physics:    {corr_phys:+.3f}")
print(f"  Pose jump:  {corr_jump:+.3f}")
print(f"  Combined:   {corr_combined:+.3f}  (wp={w_phys:.2f}, wj={w_jump:.2f})")