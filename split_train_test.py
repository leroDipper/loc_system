#!/usr/bin/env python3
"""
Script to split images into train and test folders.
Usage: python split_train_test.py <source_folder> <num_test_images>
"""

import os
import sys
import shutil
from pathlib import Path
import random

def split_train_test(source_folder, num_test_images, dry_run=True, random_split=False, seed=42):
    """
    Split images from source folder into train and test subfolders.
    
    Args:
        source_folder: Path to folder containing all images
        num_test_images: Number of images to put in test set
        dry_run: If True, only show what would be done without actually moving files
        random_split: If True, randomly select test images. If False, take evenly spaced images
        seed: Random seed for reproducibility (only used if random_split=True)
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    source = Path(source_folder)
    
    if not source.exists():
        print(f"Error: Folder '{source_folder}' does not exist")
        return
    
    if not source.is_dir():
        print(f"Error: '{source_folder}' is not a directory")
        return
    
    # Get all image files
    image_files = []
    for file in source.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    # Sort files by name
    image_files.sort(key=lambda x: x.name)
    
    if not image_files:
        print(f"No image files found in '{source_folder}'")
        return
    
    total_images = len(image_files)
    print(f"Found {total_images} image files")
    
    if num_test_images >= total_images:
        print(f"Error: Number of test images ({num_test_images}) must be less than total images ({total_images})")
        return
    
    num_train_images = total_images - num_test_images
    
    print(f"Train set: {num_train_images} images")
    print(f"Test set: {num_test_images} images")
    print()
    
    # Create train and test folders
    train_folder = source / "large_set_train"
    test_folder = source / "large_set_test"
    
    # Select test images
    if random_split:
        random.seed(seed)
        test_indices = set(random.sample(range(total_images), num_test_images))
        print(f"Using random split (seed={seed})")
    else:
        # Evenly spaced selection
        step = total_images / num_test_images
        test_indices = set(int(i * step) for i in range(num_test_images))
        print("Using evenly spaced split")
    
    print()
    
    train_files = []
    test_files = []
    
    for idx, file in enumerate(image_files):
        if idx in test_indices:
            test_files.append(file)
        else:
            train_files.append(file)
    
    if dry_run:
        print("DRY RUN - No files will be moved. Preview of split:")
        print("=" * 80)
        print(f"\nTRAIN FOLDER ({len(train_files)} images):")
        print("-" * 80)
        for file in train_files[:10]:  # Show first 10
            print(f"  {file.name}")
        if len(train_files) > 10:
            print(f"  ... and {len(train_files) - 10} more")
        
        print(f"\nTEST FOLDER ({len(test_files)} images):")
        print("-" * 80)
        for file in test_files[:10]:  # Show first 10
            print(f"  {file.name}")
        if len(test_files) > 10:
            print(f"  ... and {len(test_files) - 10} more")
        
        print("=" * 80)
        print(f"\nFolders to be created:")
        print(f"  {train_folder}")
        print(f"  {test_folder}")
        print("\nTo actually perform the split, run with --execute flag")
    else:
        # Create directories
        print("Creating directories...")
        train_folder.mkdir(exist_ok=True)
        test_folder.mkdir(exist_ok=True)
        print(f"✓ Created {train_folder}")
        print(f"✓ Created {test_folder}")
        print()
        
        # Move files
        print("Moving files to train folder...")
        for file in train_files:
            dest = train_folder / file.name
            shutil.move(str(file), str(dest))
        print(f"✓ Moved {len(train_files)} files to train folder")
        
        print("\nMoving files to test folder...")
        for file in test_files:
            dest = test_folder / file.name
            shutil.move(str(file), str(dest))
        print(f"✓ Moved {len(test_files)} files to test folder")
        
        print(f"\n✓ Successfully split {total_images} images!")
        print(f"  Train: {train_folder} ({len(train_files)} images)")
        print(f"  Test: {test_folder} ({len(test_files)} images)")

def main():
    if len(sys.argv) < 3:
        print("Usage: python split_train_test.py <source_folder> <num_test_images> [--execute] [--random]")
        print("\nOptions:")
        print("  --execute    Actually perform the split (default is dry-run)")
        print("  --random     Randomly select test images (default is evenly spaced)")
        print("\nExample:")
        print("  python split_train_test.py /path/to/images 45 --execute")
        sys.exit(1)
    
    source_folder = sys.argv[1]
    
    try:
        num_test_images = int(sys.argv[2])
    except ValueError:
        print(f"Error: num_test_images must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)
    
    execute = '--execute' in sys.argv
    random_split = '--random' in sys.argv
    
    if execute:
        response = input(f"This will split images into train/test folders. Continue? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    split_train_test(source_folder, num_test_images, dry_run=not execute, random_split=random_split)

if __name__ == "__main__":
    main()