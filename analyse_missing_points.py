#!/usr/bin/env python3
"""
Analyze which 3D points were lost in train/test split.
"""

from pathlib import Path
import numpy as np


def load_points_from_colmap(points3d_path):
    """Load 3D points with their tracks from COLMAP."""
    points = {}
    
    with open(points3d_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                
                # Parse track: (IMAGE_ID, POINT2D_IDX) pairs
                track = []
                for i in range(8, len(parts), 2):
                    if i + 1 < len(parts):
                        img_id = int(parts[i])
                        track.append(img_id)
                
                points[point_id] = track
    
    return points


def load_image_names(images_txt_path):
    """Load image ID to name mapping."""
    images = {}
    
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) == 10:
                img_id = int(parts[0])
                img_name = parts[9]
                images[img_id] = img_name
    
    return images


def main():
    map_root = Path('colmap_database/large_map')
    
    # Load COLMAP data
    print("Loading COLMAP points and images...")
    points = load_points_from_colmap(map_root / 'project_files_large_map/points3D.txt')
    images = load_image_names(map_root / 'project_files_large_map/images.txt')
    
    print(f"Total 3D points in COLMAP: {len(points)}")
    print(f"Total images in COLMAP: {len(images)}")
    
    # Identify train and test images
    train_images = set((map_root / 'large_set_train').glob('*.jpg'))
    train_names = {img.name for img in train_images}
    
    test_images = set((map_root / 'large_set_test').glob('*.jpg'))
    test_names = {img.name for img in test_images}
    
    print(f"\nTrain images: {len(train_names)}")
    print(f"Test images: {len(test_names)}")
    
    # Analyze each point
    only_in_test = 0
    only_in_train = 0
    in_both = 0
    
    for point_id, track_img_ids in points.items():
        # Get image names for this track
        track_names = [images[img_id] for img_id in track_img_ids if img_id in images]
        
        in_train = any(name in train_names for name in track_names)
        in_test = any(name in test_names for name in track_names)
        
        if in_train and in_test:
            in_both += 1
        elif in_train:
            only_in_train += 1
        elif in_test:
            only_in_test += 1
    
    print(f"\n=== Point Visibility Analysis ===")
    print(f"Points seen in BOTH train and test: {in_both}")
    print(f"Points seen ONLY in train:          {only_in_train}")
    print(f"Points seen ONLY in test:           {only_in_test}")
    
    print(f"\n=== Expected Map Sizes ===")
    print(f"Full map (all images):       {len(points)} points")
    print(f"Train map (should have):     {in_both + only_in_train} points")
    print(f"Lost due to test-only:       {only_in_test} points")
    
    # Load actual map to compare
    train_map = np.load(map_root / 'colmap_map_train_set.npz')
    actual_train_points = len(train_map['xyz_world'])
    
    print(f"\n=== Actual vs Expected ===")
    print(f"Expected train map size: {in_both + only_in_train}")
    print(f"Actual train map size:   {actual_train_points}")
    print(f"Unexplained loss:        {(in_both + only_in_train) - actual_train_points}")


if __name__ == "__main__":
    main()