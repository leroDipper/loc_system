"""
Analyze coverage statistics of the original (unpruned) map.
This helps us choose appropriate b_cover parameter for pruning.
"""

import numpy as np
from pathlib import Path

def analyze_map_coverage(map_npz_path, colmap_path):
    """
    Analyze how many points are visible per frame in the original map.
    """
    
    # Load map
    print(f"Loading map from {map_npz_path}")
    data = np.load(map_npz_path)
    xyz = data['xyz_world']
    n_points = len(xyz)
    print(f"  Total points: {n_points}")
    
    # Build co-visibility
    print(f"\nLoading COLMAP tracks from {colmap_path}")
    colmap_path = Path(colmap_path)
    
    # Load points3D.txt
    points = []
    with open(colmap_path / "points3D.txt", 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                # Parse track
                track = []
                for i in range(8, len(parts), 2):
                    if i + 1 < len(parts):
                        img_id = int(parts[i])
                        kp_idx = int(parts[i+1])
                        track.append((img_id, kp_idx))
                
                points.append({'track': track})
    
    # Load images.txt
    images_data = {}
    with open(colmap_path / "images.txt", 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) == 10:
                img_id = int(parts[0])
                img_name = parts[9]
                images_data[img_id] = img_name
    
    # Build co-visibility (map_idx -> list of img_ids)
    co_visibility = {}
    map_idx = 0
    
    for point in points:
        if not point['track']:
            continue
        
        first_img_id, first_kp_idx = point['track'][0]
        img_name = images_data.get(first_img_id)
        
        if not img_name:
            continue
        
        if map_idx >= n_points:
            break
        
        track_img_ids = [img_id for img_id, kp_idx in point['track']]
        co_visibility[map_idx] = track_img_ids
        map_idx += 1
    
    print(f"  Mapped {len(co_visibility)} points")
    
    # Get all training images
    all_images = set()
    for img_ids in co_visibility.values():
        all_images.update(img_ids)
    training_images = sorted(all_images)
    print(f"  Found {len(training_images)} training images")
    
    # Compute coverage per frame
    print(f"\nComputing coverage per frame...")
    frame_coverage = {img_id: 0 for img_id in training_images}
    
    for point_idx in range(n_points):
        if point_idx in co_visibility:
            for img_id in co_visibility[point_idx]:
                if img_id in frame_coverage:
                    frame_coverage[img_id] += 1
    
    coverage_values = list(frame_coverage.values())
    
    # Statistics
    print(f"\n{'='*60}")
    print(f"ORIGINAL MAP COVERAGE STATISTICS")
    print(f"{'='*60}")
    print(f"Total map points: {n_points}")
    print(f"Total frames: {len(training_images)}")
    print(f"\nPoints visible per frame:")
    print(f"  Mean:   {np.mean(coverage_values):.1f}")
    print(f"  Median: {np.median(coverage_values):.1f}")
    print(f"  Std:    {np.std(coverage_values):.1f}")
    print(f"  Min:    {np.min(coverage_values)}")
    print(f"  Max:    {np.max(coverage_values)}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(coverage_values, p)
        print(f"  {p:2d}th: {val:.1f}")
    
    # Distribution
    print(f"\nDistribution:")
    bins = [0, 50, 100, 150, 200, 300, 500, 1000, 10000]
    for i in range(len(bins)-1):
        count = sum(1 for v in coverage_values if bins[i] <= v < bins[i+1])
        pct = count / len(coverage_values) * 100
        print(f"  {bins[i]:4d}-{bins[i+1]:4d} points: {count:3d} frames ({pct:5.1f}%)")
    
    # Recommendations
    print(f"\n{'='*60}")
    print(f"RECOMMENDATIONS FOR b_cover")
    print(f"{'='*60}")
    
    mean_cov = np.mean(coverage_values)
    median_cov = np.median(coverage_values)
    
    print(f"\nFor 50% reduction (pruning to ~{n_points//2} points):")
    print(f"  - Conservative: b_cover = {int(mean_cov * 0.5)} (50% of mean)")
    print(f"  - Moderate:     b_cover = {int(mean_cov * 0.75)} (75% of mean)")
    print(f"  - Aggressive:   b_cover = {int(mean_cov)} (100% of mean)")
    
    print(f"\nCurrent map mean coverage is {mean_cov:.1f}")
    print(f"To maintain similar coverage after 50% reduction,")
    print(f"use b_cover ≈ {int(mean_cov * 0.5)} to {int(mean_cov * 0.75)}")


if __name__ == "__main__":
    MAP_NPZ = 'resources/tum_fr3/map_databases/tumfr3_map_train.npz'
    COLMAP_DIR = 'resources/tum_fr3/project_files'
    
    analyze_map_coverage(MAP_NPZ, COLMAP_DIR)
