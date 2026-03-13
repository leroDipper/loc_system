import pandas as pd
import numpy as np

environments = ['mh_01', 'mh_03', 'mh_05', 'tum_fr1', 'tum_fr2', 'tum_fr3', 'lab']

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
                if len(parts) == 7:
                    # EuRoC format: image_name gt_timestamp time_diff_ms error_meters rx ry rz
                    align_rows.append({
                        'frame': parts[0],
                        'alignment_error_m': float(parts[3]),
                        'rx': float(parts[4]),
                        'ry': float(parts[5]),
                        'rz': float(parts[6])
                    })
                elif len(parts) == 5:
                    # AprilTag format: image_name error_meters rx ry rz
                    align_rows.append({
                        'frame': parts[0],
                        'alignment_error_m': float(parts[1]),
                        'rx': float(parts[2]),
                        'ry': float(parts[3]),
                        'rz': float(parts[4])
                    })
    

        align_df = pd.DataFrame(align_rows)
        merged = loc_df.merge(align_df, on='frame', how='inner')

        phys = merged['translation_std_total_m']
        residuals = merged[['rx', 'ry', 'rz']].values

        # Principal axis projection
        cov = np.cov(residuals.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        principal_axis = eigenvectors[:, -1]
        merged['projection'] = np.abs(residuals @ principal_axis)

        corr_phys = np.corrcoef(phys, merged['error_m'])[0, 1]
        corr_align = np.corrcoef(merged['alignment_error_m'], merged['error_m'])[0, 1]
        corr_proj = np.corrcoef(merged['projection'], merged['error_m'])[0, 1]

        # Use projection if positive, else raw alignment if positive, else physics only
        if corr_proj > 0:
            signal = merged['projection']
        elif corr_align > 0:
            signal = merged['alignment_error_m']
        else:
            signal = phys * 0

        if signal.var() > 0:
            w = signal.var() / (signal.var() + phys.var())
        else:
            w = 0.0

        phys_norm = (phys - phys.mean()) / (phys.std() + 1e-9)
        signal_norm = (signal - signal.mean()) / (signal.std() + 1e-9)

        combined = (1 - w) * phys_norm + w * signal_norm
        corr_weighted = np.corrcoef(combined, merged['error_m'])[0, 1]

        print(f"\n{env.upper()} (n={len(merged)}, w={w:.3f})")
        print(f"  Physics:          {corr_phys:+.3f}")
        print(f"  Alignment raw:    {corr_align:+.3f}")
        print(f"  Projection:       {corr_proj:+.3f}")
        print(f"  Weighted combined:{corr_weighted:+.3f}")

    except FileNotFoundError as e:
        print(f"\n{env.upper()}: skipped — {e}")