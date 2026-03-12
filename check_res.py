import numpy as np

environments = ['mh_01', 'mh_03', 'mh_05']

# Known physics correlations for reference
physics_corr = {'mh_01': 0.610, 'mh_03': 0.298, 'mh_05': 0.456}

for env in environments:
    align_path = f'resources/{env}/project_files/alignment_results.txt'

    rows = []
    with open(align_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 7:
                rows.append([float(parts[4]), float(parts[5]), float(parts[6])])

    residuals = np.array(rows)  # (N, 3)

    cov = np.cov(residuals.T)   # 3x3 covariance matrix
    trace = np.trace(cov)
    eigenvalues = np.linalg.eigvalsh(cov)
    anisotropy = eigenvalues[-1] / (eigenvalues[0] + 1e-9)  # max / min eigenvalue
    

    print(f"\n{env.upper()}")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    principal_axis = eigenvectors[:, -1]
    print(f"  Principal axis:   x={principal_axis[0]:.3f}, y={principal_axis[1]:.3f}, z={principal_axis[2]:.3f}")
    print(f"  N residuals:      {len(residuals)}")
    print(f"  Mean error:       {np.linalg.norm(residuals, axis=1).mean()*100:.2f} cm")
    print(f"  Trace(Cov):       {trace:.6f}")
    print(f"  Anisotropy:       {anisotropy:.2f}")
    print(f"  Eigenvalues:      {eigenvalues}")
    print(f"  Physics corr:     {physics_corr[env]:.3f}")


