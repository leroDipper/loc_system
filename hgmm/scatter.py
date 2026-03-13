import pandas as pd
import numpy as np

environments = ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3']

for env in environments:
    csv_path = f'results/{env}_uncertainty250.csv'
    align_path = f'resources/{env}/project_files/alignment_results.txt'

    try:
        loc_df = pd.read_csv(csv_path)

        align_rows = []
        with open(align_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                parts = line.strip().split()
                align_rows.append({'frame': parts[0], 'alignment_error_m': float(parts[3])})

        align_df = pd.DataFrame(align_rows)
        merged = loc_df.merge(align_df, on='frame', how='inner')

        phys = merged['translation_std_total_m']
        align = merged['alignment_error_m']

        corr_phys = np.corrcoef(phys, merged['error_m'])[0, 1]
        corr_align = np.corrcoef(align, merged['error_m'])[0, 1]

        # Weight from original variances, zero if alignment is negatively correlated
        if corr_align > 0:
            residual_variance = align.var()
            physics_variance = phys.var()
            w = residual_variance / (residual_variance + physics_variance)
        else:
            w = 0.0

        # Normalise both signals to same scale
        phys_norm = (phys - phys.mean()) / (phys.std() + 1e-9)
        align_norm = (align - align.mean()) / (align.std() + 1e-9)

        combined = (1 - w) * phys_norm + w * align_norm
        corr_weighted = np.corrcoef(combined, merged['error_m'])[0, 1]

        print(f"\n{env.upper()} (n={len(merged)}, w={w:.3f})")
        print(f"  Physics:          {corr_phys:+.3f}")
        print(f"  Alignment:        {corr_align:+.3f}")
        print(f"  Weighted combined:{corr_weighted:+.3f}")

    except FileNotFoundError as e:
        print(f"\n{env.upper()}: skipped — {e}")