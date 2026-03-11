import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataObjects import LocalisationDataset, MetaLocalisationDataset

def load_all_environments(base_features, keypoints=[250, 350, 550]):

    sequences = {
        'mh_01': 'results/mh_01_uncertainty{}.csv',
        'mh_03': 'results/mh_03_uncertainty{}.csv',
        'mh_05': 'results/mh_05_uncertainty{}.csv',
        'tum_fr1': 'results/tum_fr1_uncertainty{}.csv',
        'tum_fr2': 'results/tum_fr2_uncertainty{}.csv',
        'tum_fr3': 'results/tum_fr3_uncertainty{}.csv',
        'lab':    'results/lab_uncertainty{}.csv',
    }

    train_sequences = ['mh_03', 'mh_05']
    test_sequences  = ['mh_01']

    def load_sequence(name, path_template, keypoints):
        dfs = []
        for kp in keypoints:
            path = path_template.format(kp)
            if os.path.exists(path):
                dfs.append(pd.read_csv(path))
        if not dfs:
            return None
        return pd.concat(dfs, ignore_index=True)

    def apply_transforms(df):
        df = df.copy()
        df['log_mean_inlier_track_length']   = np.log1p(df['mean_inlier_track_length'])
        df['log_mean_inverse_depth']         = np.log1p(df['mean_inverse_depth'])
        df['log_median_inlier_reproj_error'] = np.log1p(df['median_inlier_reproj_error'])
        df['log_img_blur_score']             = np.log1p(df['img_blur_score'])
        return df

    log_features = [
        "log_mean_inlier_track_length",
        "condition_estimate",
        "match_std_x",
        "depth_mean",
        "depth_std",
        "img_contrast",
        "depth_range",
        "img_brightness",
        "match_spread_normalized",
        "log_mean_inverse_depth",
        "log_median_inlier_reproj_error",
        "log_img_blur_score",
        "mean_inlier_ba_error",
    ]

    temporal_features = ['rolling_mean_reproj_error', 'rolling_std_reproj_error', 'pose_jump']

    # Load and transform all sequences
    raw = {}
    for name, path_template in sequences.items():
        df = load_sequence(name, path_template, keypoints)
        if df is None:
            print(f"  Skipping {name} — no files found")
            continue
        df = apply_transforms(df)

        # Outlier filtering
        mean_e = df['error_m'].mean()
        std_e  = df['error_m'].std()
        df = df[df['error_m'] < mean_e + 3 * std_e].copy()

        raw[name] = df
        n_fail = (df['error_m'] > 0.10).mean() * 100
        print(f"  Loaded {name}: {len(df)} frames, mean={df['error_m'].mean()*100:.1f}cm, {n_fail:.1f}% failures")

    # Determine available features
    sample_df = next(iter(raw.values()))
    available_temporal = [f for f in temporal_features if f in sample_df.columns]
    features = log_features + available_temporal
    print(f"\nUsing {len(features)} features ({len(available_temporal)} temporal)")

    # Fit scaler on training sequences only
    train_dfs = [raw[name] for name in train_sequences if name in raw]
    all_train = pd.concat(train_dfs, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(all_train[features].values)

    def build_env(name, df):
        X = scaler.transform(df[features].values)
        X = np.clip(X, -5, 5)
        y = df['error_m'].values
        return LocalisationDataset(name, X, y)

    train_envs = [build_env(name, raw[name]) for name in train_sequences if name in raw]
    test_envs  = [build_env(name, raw[name]) for name in test_sequences  if name in raw]

    meta_dataset = MetaLocalisationDataset(train_envs)

    return meta_dataset, test_envs, scaler, features