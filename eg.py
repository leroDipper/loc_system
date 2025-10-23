#!/usr/bin/env python3
"""
Example: Using the localisation package programmatically.
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
            
            # Image line has: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
            if len(parts) == 10:
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                img_name = parts[9]
                
                # Convert quaternion to rotation matrix
                R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                t = np.array([tx, ty, tz])
                
                # Camera center: C = -R^T @ t
                C = -R.T @ t
                
                colmap_poses[img_name] = C
    
    return colmap_poses

def main():
    map_root = 'colmap_database/large_map/'
    # Load map
    print("Loading map...")
    xyz_world, map_descriptors = MapLoader.load_map(map_root + 'colmap_map_large_set.npz')
    print(f"Loaded {len(xyz_world)} 3D points")

    # Dataset root path #
    dataset_root = 'colmap_database/large_map/large_set/'

    K = MapLoader.load_camera_intrinsics('colmap_database/figure_8_map/figure8/ground_truth_poses.json')
    
    gt_positions = load_colmap_ground_truth(map_root +'project_files_large_map/images.txt')
    
    # Initialize localiser
    localiser = Localiser(xyz_world, map_descriptors, K)
    
    
    
    # localise a single image
    print("\nlocalising frame_0132.jpg...")
    # result, error = localiser.localise(dataset_root + 'frame_fig8_0012.jpg')
    result, error = localiser.localise('colmap_database/large_map/large_set/frame_0132.jpg')

    
    if result:
        gt = gt_positions['frame_0132.jpg']
        error_dist = np.linalg.norm(result['position'] - gt)
        
        print(f"✓ Success!")
        print(f"  Estimated:     {result['position']}")
        print(f"  COLMAP GT:     {gt}")
        print(f"  Error:         {error_dist:.3f} meters")
        print(f"  Inliers:       {result['inliers']}/{result['total']}")

    else:
        print(f"Failed: {error}")
    
    # # localise multiple images
    # print("\nlocalising batch of images...")

    # images = [
    #     dataset_root + 'frame_0035.jpg',
    #     dataset_root + 'frame_0070.jpg'
    # ]

    
    # results = localiser.localise_batch(images)
    
    # for img_path, (result, error) in zip(images, results):
    #     img_name = img_path.split('/')[-1]
        
    #     if result:
    #         gt = gt_positions[img_name]
    #         error_dist = np.linalg.norm(result['position'] - gt)
    #         print(f"{img_name}: Error {error_dist:.3f}m | Inliers {result['inliers']}/{result['total']}")
    #     else:
    #         print(f"{img_name}: FAILED - {error}")


if __name__ == "__main__":
    main()