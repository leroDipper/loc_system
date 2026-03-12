#!/usr/bin/env python3
"""
Align COLMAP reconstruction to AprilTag ground truth using Sim(3) transformation.
Uses image filenames for matching (e.g., 000000.jpg).
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import json


def load_apriltag_ground_truth(npz_path):
    """
    Load ground truth from AprilTag detection NPZ file.
    
    Returns:
        dict: {image_name: position}
    """
    data = np.load(npz_path, allow_pickle=True)
    image_files = data['image_files']
    positions = data['positions']
    
    gt_data = {}
    for img_path, pos in zip(image_files, positions):
        # Extract just the filename (e.g., '000000.jpg')
        img_name = Path(img_path).name
        gt_data[img_name] = pos
    
    return gt_data


def load_colmap_poses(images_txt_path):
    """
    Load COLMAP camera positions from images.txt.
    
    Returns:
        dict: {image_name: camera_center}
    """
    colmap_poses = {}
    
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip comments and empty lines
        if line.startswith('#') or line == '':
            i += 1
            continue
        
        # Image line format: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
        parts = line.split()
        if len(parts) >= 10:
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            img_name = parts[9]
            
            # Convert quaternion to rotation matrix
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t = np.array([tx, ty, tz])
            
            # Camera center: C = -R^T @ t
            C = -R.T @ t
            
            colmap_poses[img_name] = C
        
        # Skip the next line (points line in images.txt)
        i += 2
    
    return colmap_poses


def match_poses(gt_data, colmap_poses):
    """
    Match COLMAP poses to ground truth using image filenames.
    
    Returns:
        gt_points: Nx3 array of ground truth positions
        colmap_points: Nx3 array of COLMAP positions
        matched_names: List of matched image names
    """
    gt_points = []
    colmap_points = []
    matched_names = []
    unmatched = []
    
    for img_name in gt_data.keys():
        if img_name in colmap_poses:
            gt_points.append(gt_data[img_name])
            colmap_points.append(colmap_poses[img_name])
            matched_names.append(img_name)
        else:
            unmatched.append(img_name)
    
    if unmatched:
        print(f"\nWarning: {len(unmatched)} ground truth images not found in COLMAP:")
        for name in unmatched[:5]:  # Show first 5
            print(f"  - {name}")
        if len(unmatched) > 5:
            print(f"  ... and {len(unmatched) - 5} more")
    
    return np.array(gt_points), np.array(colmap_points), matched_names


def compute_similarity_transform(P, Q):
    """
    Compute Sim(3) transformation from P to Q using Umeyama algorithm.
    
    Args:
        P: (N, 3) source points (COLMAP)
        Q: (N, 3) target points (ground truth)
    
    Returns:
        s: scale
        R: 3x3 rotation matrix
        t: 3x1 translation vector
    """
    N = P.shape[0]
    
    # Center the points
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute scale
    var_P = np.sum(P_centered ** 2) / N
    
    # Covariance matrix
    H = P_centered.T @ Q_centered / N
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation matrix
    R = Vt.T @ U.T
    
    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    scale = np.sum(S) / var_P
    
    # Translation
    t = centroid_Q - scale * R @ centroid_P
    
    return scale, R, t


def apply_transform(points, scale, R, t):
    """Apply Sim(3) transformation."""
    return scale * (points @ R.T) + t


def main():
    print("=" * 70)
    print("COLMAP TO APRILTAG GROUND TRUTH ALIGNMENT")
    print("=" * 70)
    
    # Paths - adjust these to your setup
    gt_npz_path = 'resources/lab/ground_truth_cleaned.npz'
    colmap_images_txt = 'resources/lab/project_files/images.txt'
    output_dir = 'resources/lab/project_files'
    
    # Load data
    print("\n📍 Loading AprilTag ground truth...")
    gt_data = load_apriltag_ground_truth(gt_npz_path)
    print(f"   Loaded {len(gt_data)} ground truth poses")
    
    print("\n🎥 Loading COLMAP reconstruction...")
    colmap_poses = load_colmap_poses(colmap_images_txt)
    print(f"   Loaded {len(colmap_poses)} COLMAP poses")
    
    # Match frames
    print("\n🔗 Matching frames by filename...")
    gt_points, colmap_points, matched_names = match_poses(gt_data, colmap_poses)
    print(f"   Found {len(matched_names)} matching frames")
    
    if len(matched_names) < 3:
        print("\n❌ Error: Need at least 3 matching frames for alignment!")
        print("   Make sure COLMAP reconstructed your images successfully.")
        return
    
    # Compute transformation
    print("\n🔄 Computing Sim(3) transformation...")
    scale, R, t = compute_similarity_transform(colmap_points, gt_points)
    
    print("\n" + "=" * 70)
    print("TRANSFORMATION PARAMETERS")
    print("=" * 70)
    print(f"\nScale factor: {scale:.6f}")
    print(f"  → 1 COLMAP unit = {scale:.4f} meters")
    
    print(f"\nRotation matrix:")
    print(R)
    
    euler_angles = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    print(f"\nRotation (Euler angles, degrees): {euler_angles}")
    
    print(f"\nTranslation vector (meters):")
    print(f"  [X: {t[0]:7.4f}, Y: {t[1]:7.4f}, Z: {t[2]:7.4f}]")
    
    # Apply transformation and compute errors
    print("\n" + "=" * 70)
    print("ALIGNMENT QUALITY")
    print("=" * 70)
    
    colmap_aligned = apply_transform(colmap_points, scale, R, t)
    residual_vectors = colmap_aligned - gt_points
    errors = np.linalg.norm(residual_vectors, axis=1)
    
    print(f"\nPosition errors after alignment:")
    print(f"  Mean:   {np.mean(errors):.4f} m  ({np.mean(errors)*100:.2f} cm)")
    print(f"  Median: {np.median(errors):.4f} m  ({np.median(errors)*100:.2f} cm)")
    print(f"  Std:    {np.std(errors):.4f} m")
    print(f"  Max:    {np.max(errors):.4f} m  ({np.max(errors)*100:.2f} cm)")
    print(f"  Min:    {np.min(errors):.4f} m  ({np.min(errors)*100:.2f} cm)")
    
    # Percentiles
    print(f"\nError percentiles:")
    print(f"  50th: {np.percentile(errors, 50):.4f} m")
    print(f"  75th: {np.percentile(errors, 75):.4f} m")
    print(f"  90th: {np.percentile(errors, 90):.4f} m")
    print(f"  95th: {np.percentile(errors, 95):.4f} m")
    
    # Show best and worst
    worst_idx = np.argmax(errors)
    best_idx = np.argmin(errors)
    
    print(f"\n📉 Best frame:  {matched_names[best_idx]} (error: {errors[best_idx]*100:.2f} cm)")
    print(f"📈 Worst frame: {matched_names[worst_idx]} (error: {errors[worst_idx]*100:.2f} cm)")
    
    # Save transformation
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    transform_data = {
        'scale': float(scale),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'num_frames_matched': len(matched_names),
        'alignment_errors': {
            'mean_meters': float(np.mean(errors)),
            'median_meters': float(np.median(errors)),
            'std_meters': float(np.std(errors)),
            'max_meters': float(np.max(errors)),
            'min_meters': float(np.min(errors))
        }
    }
    
    transform_path = Path(output_dir) / 'colmap_to_apriltag_transform.json'
    with open(transform_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    print(f"\n Saved transformation to: {transform_path}")
    
    # Save detailed results
    results_path = Path(output_dir) / 'alignment_results.txt'
    with open(results_path, 'w') as f:
        f.write("# COLMAP to AprilTag Ground Truth Alignment\n")
        f.write(f"# Ground truth: {gt_npz_path}\n")
        f.write(f"# COLMAP: {colmap_images_txt}\n")
        f.write(f"#\n")
        f.write(f"# Scale: {scale:.6f}\n")
        f.write(f"# Mean error: {np.mean(errors):.4f} m\n")
        f.write(f"# Median error: {np.median(errors):.4f} m\n")
        f.write(f"#\n")
        f.write("# Format: image_name error_meters rx ry rz\n")
        
        for name, error, rv in sorted(zip(matched_names, errors, residual_vectors), key=lambda x: x[1]):
            f.write(f"{name} {error:.6f} {rv[0]:.4f} {rv[1]:.4f} {rv[2]:.4f}\n")
    
    print(f"✅ Saved detailed results to: {results_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Successfully aligned {len(matched_names)} frames")
    print(f"✓ Average error: {np.mean(errors)*100:.2f} cm")
    print(f"✓ Median error:  {np.median(errors)*100:.2f} cm")
    print(f"✓ Scale factor:  {scale:.4f} (1 COLMAP unit = {scale:.3f} m)")
    
    # Quality assessment
    if np.mean(errors) < 0.05:
        print("\n🎉 Excellent alignment! (mean error < 5 cm)")
    elif np.mean(errors) < 0.10:
        print("\n✅ Good alignment (mean error < 10 cm)")
    elif np.mean(errors) < 0.20:
        print("\n⚠️  Fair alignment (mean error < 20 cm)")
    else:
        print("\n❌ Poor alignment - consider checking:")
        print("   - AprilTag positions accuracy")
        print("   - COLMAP reconstruction quality")
        print("   - Camera calibration")
    
    print()


if __name__ == "__main__":
    main()