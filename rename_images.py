#!/usr/bin/env python3
"""
Script to rename all images in a folder to sequential frame numbers.
Usage: python rename_images.py <folder_path>
"""

import os
import sys
from pathlib import Path

def rename_images_sequentially(folder_path, dry_run=True):
    """
    Rename all image files in folder to frame_0001.jpg, frame_0002.jpg, etc.
    
    Args:
        folder_path: Path to the folder containing images
        dry_run: If True, only show what would be renamed without actually renaming
    """
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory")
        return
    
    # Get all image files
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    # Sort files by name to ensure consistent ordering
    image_files.sort(key=lambda x: x.name)
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} image files")
    print()
    
    # Determine padding width based on number of files
    num_digits = len(str(len(image_files)))
    # Use minimum of 4 digits for padding
    num_digits = max(4, num_digits)
    
    # Keep track of the file extension from the first file (or use .jpg as default)
    default_extension = image_files[0].suffix if image_files else '.jpg'
    
    # Rename files
    rename_pairs = []
    for idx, old_path in enumerate(image_files, start=1):
        new_name = f"frame_{idx:0{num_digits}d}{old_path.suffix}"
        new_path = folder / new_name
        rename_pairs.append((old_path, new_path))
    
    # Check for conflicts
    conflicts = []
    for old_path, new_path in rename_pairs:
        if new_path.exists() and new_path != old_path:
            conflicts.append(new_path.name)
    
    if conflicts:
        print("WARNING: The following target filenames already exist:")
        for name in conflicts:
            print(f"  - {name}")
        print("\nThis might cause issues. Consider backing up your files first.")
        print()
    
    if dry_run:
        print("DRY RUN - No files will be renamed. Preview of changes:")
        print("-" * 80)
        for old_path, new_path in rename_pairs:
            print(f"{old_path.name:50s} -> {new_path.name}")
        print("-" * 80)
        print(f"\nTotal files to rename: {len(rename_pairs)}")
        print("\nTo actually rename the files, run with --execute flag")
    else:
        # Use temporary names to avoid conflicts during renaming
        temp_pairs = []
        
        # First pass: rename to temporary names
        print("Renaming files...")
        for old_path, new_path in rename_pairs:
            temp_name = f"_temp_{old_path.name}"
            temp_path = folder / temp_name
            old_path.rename(temp_path)
            temp_pairs.append((temp_path, new_path))
        
        # Second pass: rename to final names
        for temp_path, new_path in temp_pairs:
            temp_path.rename(new_path)
            print(f"✓ {new_path.name}")
        
        print(f"\n✓ Successfully renamed {len(rename_pairs)} files!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python rename_images.py <folder_path> [--execute]")
        print("\nOptions:")
        print("  --execute    Actually perform the renaming (default is dry-run)")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    execute = '--execute' in sys.argv
    
    if execute:
        response = input("This will rename all images in the folder. Are you sure? (yes/no): ")
        if response.lower() != 'yes':
            print("Aborted.")
            return
    
    rename_images_sequentially(folder_path, dry_run=not execute)

if __name__ == "__main__":
    main()