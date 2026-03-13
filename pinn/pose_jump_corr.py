import pandas as pd
import numpy as np

environments = ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3', 'lab']

def normalise(s):
    return (s - s.mean()) / (s.std() + 1e-9)

def load_alignment(align_path):
    rows = []
    with open(align_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 7:
                # EuRoC format: image_name gt_timestamp time_diff_ms error_meters rx ry rz
                rows.append({'frame': parts[0], 'alignment_error_m': float(parts[3]),
                             'rx': float(parts[4]), 'ry': float(parts[5]), 'rz': float(parts[6])})
            elif len(parts) == 5:
                # AprilTag format: image_name error_meters rx ry rz
                rows.append({'frame': parts[0], 'alignment_error_m': float(parts[1]),
                             'rx': float(parts[2]), 'ry': float(parts[3]), 'rz': float(parts[4])})
    return pd.DataFrame(rows)

all_results = []

for env in environments:
    csv_path = f'results/{env}_uncertainty250.csv'
    align_path = f'resources/{env}/project_files/alignment_results.txt'

    try:
        df = pd.read_csv(csv_path)
        df = df.sort_values('frame').reset_index(drop=True)

        # Pose jump — lab uses frame index, EuRoC/TUM use nanosecond timestamps
        positions = df[['C_x', 'C_y', 'C_z']].values
        jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        if env == 'lab':
            dt = np.ones(len(jumps))
        else:
            df['timestamp_s'] = df['frame'].str.replace('.png', '').astype(float) / 1e9
            dt = np.diff(df['timestamp_s'].values)
        velocity = jumps / (dt + 1e-9)
        df['pose_jump'] = np.concatenate([[velocity.mean()], velocity])

        if env == 'lab':
            merged = df.copy()
            merged['projection'] = 0.0
            merged['rx'] = 0.0
            merged['ry'] = 0.0
            merged['rz'] = 0.0
            w_proj = 0.0
            corr_phys = np.corrcoef(merged['translation_std_total_m'], merged['error_m'])[0, 1]
            corr_jump = np.corrcoef(merged['pose_jump'], merged['error_m'])[0, 1]
            w_phys = max(corr_phys, 0)
            w_jump = max(corr_jump, 0)
            total = w_phys + w_jump + 1e-9
            w_phys /= total; w_jump /= total
        else:
            align_df = load_alignment(align_path)
            merged = df.merge(align_df, on='frame', how='inner')

            residuals = merged[['rx', 'ry', 'rz']].values
            cov = np.cov(residuals.T)
            eigenvectors = np.linalg.eigh(cov)[1]
            principal_axis = eigenvectors[:, -1]
            merged['projection'] = np.abs(residuals @ principal_axis)

            corr_phys = np.corrcoef(merged['translation_std_total_m'], merged['error_m'])[0, 1]
            corr_proj = np.corrcoef(merged['projection'], merged['error_m'])[0, 1]
            corr_jump = np.corrcoef(merged['pose_jump'], merged['error_m'])[0, 1]

            w_phys = max(corr_phys, 0)
            w_proj = max(corr_proj, 0)
            w_jump = max(corr_jump, 0)
            total = w_phys + w_proj + w_jump + 1e-9
            w_phys /= total; w_proj /= total; w_jump /= total

        combined = (w_phys * normalise(merged['translation_std_total_m']) +
                    w_proj * normalise(merged['projection']) +
                    w_jump * normalise(merged['pose_jump']))

        pearson = np.corrcoef(combined, merged['error_m'])[0, 1]
        print(f"\n{env.upper()} (n={len(merged)})")
        print(f"  Pearson combined: {pearson:+.3f}")
        print(f"  w_phys={w_phys:.2f} w_proj={w_proj:.2f} w_jump={w_jump:.2f}")

        merged['combined_score'] = combined
        merged['w_phys'] = w_phys
        merged['w_proj'] = w_proj
        merged['w_jump'] = w_jump
        merged['physics'] = merged['translation_std_total_m']
        merged['env'] = env
        all_results.append(merged)

    except FileNotFoundError as e:
        print(f"\n{env.upper()}: skipped — {e}")

df_out = pd.concat(all_results, ignore_index=True)
df_out.to_csv('results/training_data.csv', index=False)
print(f"\nSaved training_data.csv with {len(df_out)} frames and {len(df_out.columns)} columns")