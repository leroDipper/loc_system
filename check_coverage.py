#!/usr/bin/env python3
"""
Check local coverage: how many train images are near each test image.
"""

from pathlib import Path
import re


def extract_frame_number(filename):
    """Extract frame number from filename like 'frame_0132.jpg'."""
    match = re.search(r'frame_(\d+)', filename)
    if match:
        return int(match.group(1))
    return None


def main():
    map_root = Path('colmap_database/large_map')
    
    # Get train and test images
    train_images = sorted(map_root.glob('large_set_train/*.jpg'))
    test_images = sorted(map_root.glob('large_set_test/*.jpg'))
    
    # Extract frame numbers
    train_frames = {extract_frame_number(img.name): img.name for img in train_images}
    test_frames = {extract_frame_number(img.name): img.name for img in test_images}
    
    print(f"Train images: {len(train_frames)}")
    print(f"Test images: {len(test_frames)}")
    print(f"\nFrame range: {min(train_frames.keys())} to {max(max(train_frames.keys()), max(test_frames.keys()))}")
    
    # Analyze coverage for each test image
    window = 10  # Look ±10 frames around each test image
    
    print(f"\n=== Local Coverage (±{window} frames) ===")
    
    test_coverage = []
    for test_num in sorted(test_frames.keys()):
        # Count train images within window
        nearby_train = 0
        for offset in range(-window, window + 1):
            if test_num + offset in train_frames:
                nearby_train += 1
        
        test_coverage.append((test_num, test_frames[test_num], nearby_train))
        print(f"frame_{test_num:04d}.jpg: {nearby_train}/{window*2+1} train images nearby")
    
    # Show statistics
    coverages = [count for _, _, count in test_coverage]
    print(f"\n=== Coverage Statistics ===")
    print(f"Best coverage:  {max(coverages)} train images")
    print(f"Worst coverage: {min(coverages)} train images")
    print(f"Average:        {sum(coverages)/len(coverages):.1f} train images")
    
    # Identify poorly covered test images
    print(f"\n=== Poorly Covered Test Images (< {window} nearby) ===")
    for test_num, test_name, count in test_coverage:
        if count < window:
            print(f"{test_name}: only {count} train images nearby")


if __name__ == "__main__":
    main()