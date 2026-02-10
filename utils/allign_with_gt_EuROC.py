#!/usr/bin/env python3
"""
Align COLMAP reconstruction to EuROC MAV ground truth using Sim(3) transformation.
Uses timestamp-based filenames for matching.
"""

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
import pandas as pd


def load_ground_truth(gt_path):
    """
    Load ground truth trajectory from EUROC MAV format.
    
    Format: timestamp tx ty tz qw qx qy qz
    
    Returns:
        dict: {timestamp_in_seconds: (position, quaternion)}
    """
    gt_data = {}
    df = pd.read_csv(gt_path, comment='#', header=None)
    
    timestamps = df.iloc[:, 0].values
    tx = df.iloc[:, 1].values
    ty = df.iloc[:, 2].values
    tz = df.iloc[:, 3].values
    qw = df.iloc[:, 4].values
    qx = df.iloc[:, 5].values
    qy = df.iloc[:, 6].values
    qz = df.iloc[:, 7].values

    for i in range(len(timestamps)):
        timestamp = int(timestamps[i])
        # Convert nanosecond timestamp to seconds
        timestamp_sec = timestamp / 1e9
        
        tx_, ty_, tz_ = float(tx[i]), float(ty[i]), float(tz[i])
        qx_, qy_, qz_, qw_ = float(qx[i]), float(qy[i]), float(qz[i]), float(qw[i])
        position = np.array([tx_, ty_, tz_])
        quaternion = np.array([qx_, qy_, qz_, qw_])
        gt_data[timestamp_sec] = (position, quaternion)
    return gt_data



def load_colmap_poses(images_txt_path):
    """
    Load COLMAP camera positions from images.txt.
    
    Returns:
        dict: {image_name: camera_center}
    """
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


def extract_timestamp_from_filename(filename):
    """
    Extract timestamp from TUM RGB-D filename.
    Example: '1305031523.092297.png' -> 1305031523.092297
    """
    name_without_ext = Path(filename).stem
    try:
        timestamp = float(name_without_ext)
        # If timestamp is very large (>1e12), it's likely nanoseconds
        if timestamp > 1e12:
            return timestamp / 1e9  # Convert to seconds
        return timestamp
    except ValueError:
        return None


def find_corresponding_poses(gt_data, colmap_poses, time_tolerance=0.02):
    """
    Match COLMAP images to ground truth using timestamps.
    
    Args:
        gt_data: Ground truth data {timestamp: (position, quaternion)}
        colmap_poses: COLMAP poses {image_name: camera_center}
        time_tolerance: Maximum time difference for matching (seconds)
    
    Returns:
        gt_points: Nx3 array of ground truth positions
        colmap_points: Nx3 array of COLMAP positions
        matches: List of (image_name, gt_timestamp, time_diff)
    """
    gt_timestamps = np.array(sorted(gt_data.keys()))
    
    gt_points = []
    colmap_points = []
    matches = []
    
    for img_name, colmap_pos in colmap_poses.items():
        # Extract timestamp from filename
        img_timestamp = extract_timestamp_from_filename(img_name)
        if img_timestamp is None:
            print(f"Warning: Could not extract timestamp from {img_name}")
            continue
        
        # Find closest ground truth timestamp
        time_diffs = np.abs(gt_timestamps - img_timestamp)
        closest_idx = np.argmin(time_diffs)
        closest_timestamp = gt_timestamps[closest_idx]
        time_diff = time_diffs[closest_idx]
        
        # Check if within tolerance
        if time_diff <= time_tolerance:
            gt_pos, gt_quat = gt_data[closest_timestamp]
            gt_points.append(gt_pos)
            colmap_points.append(colmap_pos)
            matches.append((img_name, closest_timestamp, time_diff))
        else:
            print(f"Warning: No close ground truth match for {img_name} (closest diff: {time_diff:.4f}s)")
    
    return np.array(gt_points), np.array(colmap_points), matches


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
    print("=" * 60)
    print("COLMAP to EuROC Ground Truth Alignment")
    print("=" * 60)
    
    # Paths (update these to your actual paths)
    gt_path = 'resources/mh_03/data.csv'
    colmap_path = 'resources/mh_03/project_files_og/images.txt'
    output_dir = 'resources/mh_03/project_files_og'
    # Load data
    print("\nLoading ground truth...")
    gt_data = load_ground_truth(gt_path)
    print(f"Loaded {len(gt_data)} ground truth poses")
    
    print("\nLoading COLMAP poses...")
    colmap_poses = load_colmap_poses(colmap_path)
    print(f"Loaded {len(colmap_poses)} COLMAP poses")
    
    # Find corresponding frames
    print("\nMatching frames (time tolerance: 0.02s)...")
    gt_points, colmap_points, matches = find_corresponding_poses(gt_data, colmap_poses, time_tolerance=0.01)
    print(f"Found {len(matches)} matching frames")
    
    if len(matches) < 3:
        print("\n❌ Error: Need at least 3 corresponding frames!")
        return
    
    # Show time matching quality
    time_diffs = [m[2] for m in matches]
    print(f"  Mean time diff: {np.mean(time_diffs)*1000:.2f} ms")
    print(f"  Max time diff:  {np.max(time_diffs)*1000:.2f} ms")
    
    # Compute transformation
    print("\nComputing Sim(3) transformation...")
    scale, R, t = compute_similarity_transform(colmap_points, gt_points)
    
    print(f"\n{'=' * 60}")
    print("TRANSFORMATION PARAMETERS")
    print(f"{'=' * 60}")
    print(f"\nScale factor: {scale:.6f}")
    print(f"  → 1 COLMAP unit = {scale:.4f} meters")
    
    print(f"\nRotation matrix:")
    print(R)
    
    print(f"\nTranslation vector:")
    print(t)
    
    # Apply transformation and compute errors
    print(f"\n{'=' * 60}")
    print("ALIGNMENT ERRORS (after transformation)")
    print(f"{'=' * 60}")
    
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
    
    print(f"\nWorst frame: {matches[worst_idx][0]} (error: {errors[worst_idx]:.4f}m)")
    print(f"Best frame:  {matches[best_idx][0]} (error: {errors[best_idx]:.4f}m)")
    
    # Save transformation
    import json
    transform_data = {
        'scale': float(scale),
        'rotation': R.tolist(),
        'translation': t.tolist(),
        'num_frames_matched': len(matches),
        'mean_alignment_error_meters': float(np.mean(errors)),
        'median_alignment_error_meters': float(np.median(errors)),
        'std_alignment_error_meters': float(np.std(errors))
    }
    
    output_path = Path(output_dir) / 'colmap_to_gt_transform.json'
    with open(output_path, 'w') as f:
        json.dump(transform_data, f, indent=2)
    
    print(f"\n✓ Saved transformation to {output_path}")
    
    # Save detailed results
    results_path = Path(output_dir) / 'alignment_results.txt'
    with open(results_path, 'w') as f:
        f.write("# COLMAP to Ground Truth Alignment Results\n")
        f.write(f"# Dataset: {gt_path}\n")
        f.write(f"# COLMAP: {colmap_path}\n")
        f.write(f"#\n")
        f.write(f"# Scale: {scale:.6f}\n")
        f.write(f"# Mean error: {np.mean(errors):.4f} m\n")
        f.write(f"# Median error: {np.median(errors):.4f} m\n")
        f.write(f"#\n")
        f.write("# image_name gt_timestamp time_diff_ms error_meters\n")
        
        for (img_name, gt_ts, time_diff), error in zip(matches, errors):
            f.write(f"{img_name} {gt_ts:.6f} {time_diff*1000:.2f} {error:.4f}\n")
    
    print(f"✓ Saved detailed results to {results_path}")
    
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Successfully aligned {len(matches)} frames")
    print(f"Average position error: {np.mean(errors)*100:.2f} cm")
    print(f"Scale factor: 1 COLMAP unit = {scale:.3f} meters")


if __name__ == "__main__":
    main()