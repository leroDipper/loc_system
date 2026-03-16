import pandas as pd
import numpy as np
from scipy.stats import rankdata

environments = ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3', 'lab']

# Features that can blow up numerically — clip at 99th percentile per environment
CLIP_FEATURES = ['condition_3d', 'trans_anisotropy', 'condition_estimate', 'pose_jump']

def load_alignment(align_path):
    rows = []
    with open(align_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 7:
                rows.append({'frame': parts[0], 'rx': float(parts[4]), 'ry': float(parts[5]), 'rz': float(parts[6]), 'error': float(parts[3])})
            elif len(parts) == 5:
                rows.append({'frame': parts[0], 'rx': float(parts[2]), 'ry': float(parts[3]), 'rz': float(parts[4]), 'error': float(parts[1])})
    return pd.DataFrame(rows)

def compute_pose_jump(df, env):
    df = df.sort_values('frame').reset_index(drop=True)
    positions = df[['C_x', 'C_y', 'C_z']].values
    jumps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    if env == 'lab':
        dt = np.ones(len(jumps))
    elif env.startswith('mh'):
        timestamps = df['frame'].str.replace('.png', '').astype(float)
        dt = np.diff(timestamps.values) / 1e9
    else:
        timestamps = df['frame'].str.replace('.png', '').astype(float)
        dt = np.diff(timestamps.values)
    velocity = jumps / (dt + 1e-9)
    df['pose_jump'] = np.concatenate([[velocity.mean()], velocity])
    return df

def rank_normalise(series):
    return rankdata(series, method='average') / len(series)

def normalise(s):
    return (s - s.mean()) / (s.std() + 1e-9)

def compute_env_features(merged, align_df=None):
    features = {}
    for col, name in [('physics', 'physics'), ('projection', 'projection')]:
        features[f'{name}_mean'] = merged[col].mean()
        features[f'{name}_std'] = merged[col].std()
    if align_df is not None and len(align_df) > 0:
        residuals = align_df[['rx', 'ry', 'rz']].values
        features['mean_alignment_error'] = align_df['error'].mean()
        cov = np.cov(residuals.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        features['anisotropy'] = eigenvalues[-1] / (eigenvalues[0] + 1e-9)
        principal_axis = eigenvectors[:, -1]
        features['principal_axis_x'] = float(principal_axis[0])
        features['principal_axis_y'] = float(principal_axis[1])
        features['principal_axis_z'] = float(principal_axis[2])
    else:
        features['mean_alignment_error'] = 0.0
        features['anisotropy'] = 1.0
        features['principal_axis_x'] = 0.0
        features['principal_axis_y'] = 0.0
        features['principal_axis_z'] = 0.0
    return features

all_dfs = []

for env in environments:
    csv_path = f'results/{env}_uncertainty250.csv'
    align_path = f'resources/{env}/project_files/alignment_results.txt'

    try:
        df = pd.read_csv(csv_path)
        df = compute_pose_jump(df, env)

        if env == 'lab':
            df['projection'] = 0.0
            merged = df.copy()
            align_df = None
        else:
            align_df = load_alignment(align_path)
            merged = df.merge(align_df, on='frame', how='inner')
            residuals = merged[['rx', 'ry', 'rz']].values
            cov = np.cov(residuals.T)
            eigenvectors = np.linalg.eigh(cov)[1]
            principal_axis = eigenvectors[:, -1]
            merged['projection'] = np.abs(residuals @ principal_axis)

        merged = merged.rename(columns={'translation_std_total_m': 'physics'})

        # Clip numerically unstable features at 99th percentile per environment
        for col in CLIP_FEATURES:
            if col in merged.columns:
                merged[col] = merged[col].clip(upper=merged[col].quantile(0.99))

        # Combined score
        corr_phys = np.corrcoef(merged['physics'], merged['error_m'])[0, 1]
        corr_proj = np.corrcoef(merged['projection'], merged['error_m'])[0, 1] if env != 'lab' else 0.0
        corr_jump = np.corrcoef(merged['pose_jump'], merged['error_m'])[0, 1]

        w_phys = max(corr_phys, 0)
        w_proj = max(corr_proj, 0)
        w_jump = max(corr_jump, 0)
        total = w_phys + w_proj + w_jump + 1e-9
        w_phys /= total; w_proj /= total; w_jump /= total

        merged['combined_score'] = (w_phys * normalise(merged['physics']) +
                                    w_proj * normalise(merged['projection']) +
                                    w_jump * normalise(merged['pose_jump']))
        merged['w_phys'] = w_phys
        merged['w_proj'] = w_proj
        merged['w_jump'] = w_jump

        pearson = np.corrcoef(merged['combined_score'], merged['error_m'])[0, 1]
        print(f"{env.upper()}: {len(merged)} frames | Pearson={pearson:+.3f} | w_phys={w_phys:.2f} w_proj={w_proj:.2f} w_jump={w_jump:.2f}")

        merged['physics_rank'] = rank_normalise(merged['physics'])
        merged['projection_rank'] = rank_normalise(merged['projection'])
        merged['pose_jump_rank'] = rank_normalise(merged['pose_jump'])

        env_feats = compute_env_features(merged, align_df)
        for k, v in env_feats.items():
            merged[k] = v

        merged['env'] = env

        keep_cols = ['frame', 'env',
                     'physics', 'projection', 'pose_jump',
                     'physics_rank', 'projection_rank', 'pose_jump_rank',
                     'physics_mean', 'physics_std',
                     'projection_mean', 'projection_std',
                     'mean_alignment_error', 'anisotropy',
                     'principal_axis_x', 'principal_axis_y', 'principal_axis_z',
                     'error_m',
                     'mean_inlier_reproj_error', 'std_inlier_reproj_error',
                     'match_spread_normalized', 'depth_relative_std', 'condition_estimate',
                     'mean_inlier_track_length', 'mean_inlier_ba_error', 'frac_high_quality_inliers',
                     'n_matches', 'n_inliers',
                     'combined_score', 'w_phys', 'w_proj', 'w_jump',
                     'reproj_error_dist', 'condition_3d', 'trans_max_std', 'trans_anisotropy', 'translation_std_crb']
        merged = merged[keep_cols]
        all_dfs.append(merged)

    except FileNotFoundError as e:
        print(f"{env.upper()}: skipped — {e}")

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv('results/training_data.csv', index=False)
print(f"\nTotal: {len(combined)} frames across {combined['env'].nunique()} environments")
print(f"Columns: {combined.columns.tolist()}")