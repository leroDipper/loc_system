#!/usr/bin/env python3
"""
Simple parameter sweep to find best settings.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from modules import MapLoader, Localiser


def load_colmap_ground_truth(images_txt_path):
    """Load ground truth camera positions from COLMAP images.txt file."""
    colmap_poses = {}
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split()
            
            if len(parts) == 10:
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                img_name = parts[9]
                
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                C = -R.T @ t
                
                colmap_poses[img_name] = C
    
    return colmap_poses


def main():
    # Load map once
    map_root = 'colmap_database/large_map/'
    print("Loading map...")
    xyz_world, map_descriptors = MapLoader.load_map(map_root + 'colmap_map_train_set.npz')
    
    K = MapLoader.load_camera_intrinsics('colmap_database/figure_8_map/figure8/ground_truth_poses.json')
    gt_positions = load_colmap_ground_truth(map_root + 'project_files_large_map/images.txt')
    
    test_image = 'colmap_database/large_map/large_set_test/frame_0132.jpg'
    gt = gt_positions['frame_0132.jpg']
    
    # Parameters to test
    ratio_thresholds = [0.70, 0.75, 0.80, 0.85]
    reprojection_errors = [6.0, 8.0, 10.0, 12.0]
    
    print(f"\nTesting {len(ratio_thresholds) * len(reprojection_errors)} parameter combinations...\n")
    
    best_error = float('inf')
    best_params = None
    
    # Test all combinations
    for ratio in ratio_thresholds:
        for reproj in reprojection_errors:
            localiser = Localiser(xyz_world, map_descriptors, K, 
                                 ratio_threshold=ratio,
                                 reprojection_error=reproj)
            
            result, error = localiser.localise(test_image)
            
            if result:
                error_dist = np.linalg.norm(result['position'] - gt)
                print(f"ratio={ratio:.2f}, reproj={reproj:4.1f} → Error: {error_dist:.3f}m, Inliers: {result['inliers']}/{result['total']}")
                
                if error_dist < best_error:
                    best_error = error_dist
                    best_params = (ratio, reproj)
            else:
                print(f"ratio={ratio:.2f}, reproj={reproj:4.1f} → FAILED")
    
    print(f"\n✓ Best parameters: ratio={best_params[0]:.2f}, reproj={best_params[1]:.1f}")
    print(f"  Best error: {best_error:.3f}m")


if __name__ == "__main__":
    main()