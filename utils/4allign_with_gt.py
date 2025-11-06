#!/usr/bin/env python3
"""
Align COLMAP reconstruction to ground truth using Sim(3) transformation.
Handles coordinate frame differences (Y/Z swap between Blender and COLMAP).
"""

import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation


def load_ground_truth(json_path):
    """Load ground truth camera positions from Blender JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    gt_poses = {}
    for pose_data in data['poses']:
        frame_num = pose_data['frame']
        frame_name = f"frame_{frame_num:04d}.jpg"
        
        # Ground truth position (Blender coordinates: Z-up)
        translation = np.array(pose_data['left_camera']['translation'])
        
        gt_poses[frame_name] = translation
    
    return gt_poses


def load_colmap_poses(images_txt_path):
    """Load COLMAP camera positions from images.txt."""
    colmap_poses = {}
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            
            parts = line.strip().split()
            
            # Image line: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
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


def find_corresponding_poses(gt_poses, colmap_poses):
    """Find frames that exist in both ground truth and COLMAP."""
    common_frames = set(gt_poses.keys()) & set(colmap_poses.keys())
    
    gt_points = []
    colmap_points = []
    
    for frame in sorted(common_frames):
        gt_points.append(gt_poses[frame])
        colmap_points.append(colmap_poses[frame])
    
    return np.array(gt_points), np.array(colmap_points), list(common_frames)


def compute_similarity_transform(P, Q):
    """
    Compute Sim(3) transformation (scale + rotation + translation) from P to Q.
    Uses Umeyama algorithm.
    
    Args:
        P: (N, 3) source points (COLMAP)
        Q: (N, 3) target points (ground truth)
    
    Returns:
        s: scale
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    assert P.shape == Q.shape
    N = P.shape[0]
    
    # Center the points
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute scale
    var_P = np.sum(P_centered ** 2) / N
    
    # Compute covariance matrix
    H = P_centered.T @ Q_centered / N
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation
    R = Vt.T @ U.T
    
    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute scale
    scale = np.sum(S) / var_P
    
    # Compute translation
    t = centroid_Q - scale * R @ centroid_P
    
    return scale, R, t


def apply_transform(points, scale, R, t):
    """Apply Sim(3) transformation to points."""
    return scale * (points @ R.T) + t


def main():
    print("="*60)
    print("COLMAP to Ground Truth Alignment")
    print("="*60)
    
    # Paths
    gt_path = 'colmap_database/large_map_xfeat/ground_truth_poses_circ_f8.json'
    colmap_path = 'colmap_database/large_map_xfeat/project_files/images.txt'
    
    # Load data
    print("\nLoading ground truth...")
    gt_poses = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_poses)} ground truth poses")
    
    print("\nLoading COLMAP poses...")
    colmap_poses = load_colmap_poses(colmap_path)
    print(f"Loaded {len(colmap_poses)} COLMAP poses")
    
    # Find corresponding frames
    print("\nFinding corresponding frames...")
    gt_points, colmap_points, common_frames = find_corresponding_poses(gt_poses, colmap_poses)
    print(f"Found {len(common_frames)} frames in both datasets")
    
    if len(common_frames) < 3:
        print("\n❌ Error: Need at least 3 corresponding frames!")
        return
    
    # Compute transformation
    print("\nComputing Sim(3) transformation...")
    scale, R, t = compute_similarity_transform(colmap_points, gt_points)
    
    print(f"\n{'='*60}")
    print("TRANSFORMATION PARAMETERS")
    print(f"{'='*60}")
    print(f"\nScale factor: {scale:.6f}")
    print(f"  → 1 COLMAP unit = {scale:.4f} meters")
    
    print(f"\nRotation matrix:")
    print(R)
    
    print(f"\nTranslation vector:")
    print(t)
    
    # Apply transformation and compute errors
    print(f"\n{'='*60}")
    print("ALIGNMENT ERRORS (after transformation)")
    print(f"{'='*60}")
    
    colmap_aligned = apply_transform(colmap_points, scale, R, t)
    errors = np.linalg.norm(colmap_aligned - gt_points, axis=1)
    
    print(f"\nMean error:   {np.mean(errors):.4f} meters")
    print(f"Median error: {np.median(errors):.4f} meters")
    print(f"Std error:    {np.std(errors):.4f} meters")
    print(f"Max error:    {np.max(errors):.4f} meters")
    print(f"Min error:    {np.min(errors):.4f} meters")
    
    # Show worst and best frames
    worst_idx = np.argmax(errors)
    best_idx = np.argmin(errors)
    
    print(f"\nWorst frame: {common_frames[worst_idx]} (error: {errors[worst_idx]:.4f}m)")
    print(f"Best frame:  {common_frames[best_idx]} (error: {errors[best_idx]:.4f}m)")
    
    # Save transformation
    transform_data = {
        'scale': float(scale),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'num_frames_used': len(common_frames),
        'mean_alignment_error_meters': float(np.mean(errors))
    }
    
    output_path = 'colmap_database/large_map_xfeat/colmap_to_gt_transform.json'
    with open(output_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    print(f"\n✓ Saved transformation to {output_path}")
    
    # Show example usage
    print(f"\n{'='*60}")
    print("HOW TO USE THIS TRANSFORMATION")
    print(f"{'='*60}")
    print("""
# To transform COLMAP positions to ground truth (meters):
scale = {:.6f}
R = np.array({})
t = np.array({})

# Transform a COLMAP position:
colmap_pos = np.array([x, y, z])  # COLMAP position
gt_pos = scale * (R @ colmap_pos) + t  # Ground truth position (meters)

# Compute error in meters:
estimated_pos = ...  # Your localization result (COLMAP units)
estimated_pos_meters = scale * (R @ estimated_pos) + t
error_meters = np.linalg.norm(estimated_pos_meters - ground_truth_meters)
""".format(scale, R.tolist(), t.tolist()))


if __name__ == "__main__":
    main()