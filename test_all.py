#!/usr/bin/env python3
"""
Test localisation on all test images and show error distribution.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from modules import MapLoader, Localiser
from pathlib import Path


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
    map_root = 'colmap_database/large_map/'
    
    # Load map
    print("Loading map...")
    xyz_world, map_descriptors = MapLoader.load_map(map_root + 'colmap_map_train_set.npz')
    print(f"Loaded {len(xyz_world)} 3D points")

    K = MapLoader.load_camera_intrinsics('colmap_database/figure_8_map/figure8/ground_truth_poses.json')
    gt_positions = load_colmap_ground_truth(map_root + 'project_files_large_map/images.txt')
    
    # Initialize localiser with best parameters
    localiser = Localiser(xyz_world, map_descriptors, K,
                         ratio_threshold=0.70,
                         reprojection_error=12.0)
    
    # Get all test images
    test_dir = Path('colmap_database/large_map/large_set_test')
    test_images = sorted(test_dir.glob('*.jpg'))
    
    print(f"\nTesting on {len(test_images)} test images...\n")
    
    # Store results
    errors = []
    failures = []
    
    for img_path in test_images:
        img_name = img_path.name
        result, error = localiser.localise(str(img_path))
        
        if result and img_name in gt_positions:
            gt = gt_positions[img_name]
            error_dist = np.linalg.norm(result['position'] - gt)
            errors.append(error_dist)
            print(f"{img_name}: {error_dist:.3f}m | Inliers: {result['inliers']}/{result['total']}")
        else:
            failures.append(img_name)
            print(f"{img_name}: FAILED")
    
    # Statistics
    print(f"\n=== Results Summary ===")
    print(f"Successful: {len(errors)}/{len(test_images)}")
    print(f"Failed: {len(failures)}/{len(test_images)}")
    
    if errors:
        errors = np.array(errors)
        print(f"\n=== Error Statistics (meters) ===")
        print(f"Mean:   {np.mean(errors):.3f}")
        print(f"Median: {np.median(errors):.3f}")
        print(f"Std:    {np.std(errors):.3f}")
        print(f"Min:    {np.min(errors):.3f}")
        print(f"Max:    {np.max(errors):.3f}")
        
        # Show distribution
        print(f"\n=== Error Distribution ===")
        bins = [0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        for i in range(len(bins)-1):
            count = np.sum((errors >= bins[i]) & (errors < bins[i+1]))
            print(f"{bins[i]:.2f}m - {bins[i+1]:.2f}m: {count} images")
        count = np.sum(errors >= bins[-1])
        print(f">= {bins[-1]:.2f}m: {count} images")


if __name__ == "__main__":
    main()