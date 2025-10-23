#!/usr/bin/env python3
"""
Simple benchmark script for measuring localization performance.
Run this on both laptop and Raspberry Pi to compare.
"""

import time
import psutil
import platform
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from modules import MapLoader, Localiser


def get_system_info():
    """Get basic system information."""
    return {
        'platform': platform.system(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(),
        'total_ram_gb': psutil.virtual_memory().total / (1024**3)
    }


def measure_memory():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024**2)


def load_colmap_ground_truth(images_txt_path):
    """Load ground truth camera positions."""
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


def benchmark_single_image(localiser, image_path, img_name, gt_positions):
    """Benchmark a single image localization with timing breakdown."""
    timings = {}
    
    # Time: Feature extraction
    start = time.time()
    try:
        keypoints, descriptors = localiser.feature_extractor.extract(image_path)
    except Exception as e:
        return None, str(e), {}
    timings['feature_extraction'] = time.time() - start
    
    if descriptors is None or len(descriptors) < 4:
        return None, "Not enough features", timings
    
    # Time: Feature matching
    start = time.time()
    matched_map_indices, matched_2d_points = localiser.matcher.match(
        localiser.map_descriptors,
        descriptors,
        keypoints
    )
    timings['matching'] = time.time() - start
    
    if matched_map_indices is None or len(matched_map_indices) < 4:
        return None, "Not enough matches", timings
    
    matched_3d_points = localiser.xyz_world[matched_map_indices]
    
    # Time: Pose estimation (RANSAC + PnP)
    start = time.time()
    pose = localiser.pose_estimator.estimate_pose(
        matched_3d_points,
        matched_2d_points,
        localiser.K
    )
    timings['pose_estimation'] = time.time() - start
    
    if pose is None:
        return None, "Pose estimation failed", timings
    
    # Calculate error if ground truth available
    if img_name in gt_positions:
        gt = gt_positions[img_name]
        error = np.linalg.norm(pose['position'] - gt)
        pose['error'] = error
    
    return pose, None, timings


def main():
    print("=" * 60)
    print("VISUAL LOCALIZATION BENCHMARK")
    print("=" * 60)
    
    # System info
    sys_info = get_system_info()
    print("\n=== System Information ===")
    print(f"Platform:     {sys_info['platform']} {sys_info['machine']}")
    print(f"Processor:    {sys_info['processor']}")
    print(f"CPU Cores:    {sys_info['cpu_count']}")
    print(f"Total RAM:    {sys_info['total_ram_gb']:.2f} GB")
    print(f"Python:       {sys_info['python_version']}")
    
    # Configuration
    map_root = 'colmap_database/large_map/'
    test_dir = Path('colmap_database/large_map/large_set_test')
    
    # Load map
    print("\n=== Map Loading ===")
    mem_before = measure_memory()
    start = time.time()
    
    xyz_world, map_descriptors = MapLoader.load_map(map_root + 'colmap_map_train_set.npz')
    K = MapLoader.load_camera_intrinsics('colmap_database/figure_8_map/figure8/ground_truth_poses.json')
    gt_positions = load_colmap_ground_truth(map_root + 'project_files_large_map/images.txt')
    
    load_time = time.time() - start
    mem_after = measure_memory()
    map_memory = mem_after - mem_before
    
    print(f"Map points:   {len(xyz_world)}")
    print(f"Load time:    {load_time:.3f}s")
    print(f"Map memory:   {map_memory:.2f} MB")
    
    # Initialize localiser
    localiser = Localiser(xyz_world, map_descriptors, K,
                         ratio_threshold=0.70,
                         reprojection_error=12.0,
                         min_inliers=15)
    
    # Get test images (use first 10 for quick benchmark)
    test_images = sorted(test_dir.glob('*.jpg'))[:10]
    print(f"\n=== Benchmarking on {len(test_images)} images ===")
    
    # Benchmark each image
    all_timings = []
    errors = []
    successes = 0
    failures = 0
    
    for img_path in test_images:
        img_name = img_path.name
        
        # Full localization with timing
        start_total = time.time()
        result, error_msg, timings = benchmark_single_image(
            localiser, str(img_path), img_name, gt_positions
        )
        total_time = time.time() - start_total
        
        if result:
            successes += 1
            timings['total'] = total_time
            all_timings.append(timings)
            if 'error' in result:
                errors.append(result['error'])
            print(f"✓ {img_name}: {total_time:.3f}s | Error: {result.get('error', 'N/A'):.3f}m")
        else:
            failures += 1
            print(f"✗ {img_name}: FAILED - {error_msg}")
    
    # Statistics
    print("\n" + "=" * 60)
    print("=== RESULTS ===")
    print("=" * 60)
    
    print(f"\nSuccess rate: {successes}/{len(test_images)} ({100*successes/len(test_images):.1f}%)")
    
    if all_timings:
        # Timing statistics
        avg_timings = {
            'feature_extraction': np.mean([t['feature_extraction'] for t in all_timings]),
            'matching': np.mean([t['matching'] for t in all_timings]),
            'pose_estimation': np.mean([t['pose_estimation'] for t in all_timings]),
            'total': np.mean([t['total'] for t in all_timings])
        }
        
        print("\n=== Average Timing (seconds) ===")
        print(f"Feature extraction: {avg_timings['feature_extraction']:.3f}s")
        print(f"Feature matching:   {avg_timings['matching']:.3f}s")
        print(f"Pose estimation:    {avg_timings['pose_estimation']:.3f}s")
        print(f"Total per image:    {avg_timings['total']:.3f}s")
        print(f"Throughput:         {1.0/avg_timings['total']:.2f} FPS")
        
        # Breakdown percentages
        print("\n=== Time Breakdown ===")
        total = avg_timings['total']
        print(f"Feature extraction: {100*avg_timings['feature_extraction']/total:.1f}%")
        print(f"Feature matching:   {100*avg_timings['matching']/total:.1f}%")
        print(f"Pose estimation:    {100*avg_timings['pose_estimation']/total:.1f}%")
    
    if errors:
        print("\n=== Localization Accuracy ===")
        print(f"Mean error:   {np.mean(errors):.3f}m")
        print(f"Median error: {np.median(errors):.3f}m")
        print(f"Std error:    {np.std(errors):.3f}m")
        print(f"Min error:    {np.min(errors):.3f}m")
        print(f"Max error:    {np.max(errors):.3f}m")
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()