#!/usr/bin/env python3
"""
Rectify (undistort) images while computing correct camera intrinsics
"""

import cv2
import numpy as np
import yaml
from pathlib import Path

# ==================== CONFIGURATION ====================
INPUT_DIR = "/home/leroy/Downloads/cam0/data"
OUTPUT_DIR = "resources/mh_03/images"
SENSOR_YAML = "/home/leroy/Downloads/cam0/sensor.yaml"
# =======================================================


def load_camera_params(yaml_path):
    """
    Load camera parameters from sensor YAML file
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract resolution
    resolution = data['resolution']  # [width, height]
    IMAGE_SIZE = tuple(resolution)
    
    # Extract intrinsics [fx, fy, cx, cy]
    intrinsics = data['intrinsics']
    K_original = np.array([
        [intrinsics[0], 0, intrinsics[2]],
        [0, intrinsics[1], intrinsics[3]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Extract distortion coefficients
    dist_coeffs = np.array(data['distortion_coefficients'], dtype=np.float64)
    
    return IMAGE_SIZE, K_original, dist_coeffs


def rectify_images():
    """
    Main function to rectify all images (remove distortion)
    """
    print("=" * 60)
    print("IMAGE RECTIFICATION")
    print("=" * 60)
    print(f"Input directory:  {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Sensor YAML:      {SENSOR_YAML}")
    print()
    
    # Load camera parameters from YAML
    print("Loading camera parameters from YAML...")
    IMAGE_SIZE, K_original, dist_coeffs = load_camera_params(SENSOR_YAML)
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print()
    
    # Step 1: Compute optimal rectification parameters
    print("Step 1: Computing rectification parameters...")
    new_K_rect, roi = cv2.getOptimalNewCameraMatrix(
        K_original, 
        dist_coeffs, 
        IMAGE_SIZE, 
        alpha=0,  # 0 = no black pixels, 1 = all pixels
        newImgSize=IMAGE_SIZE
    )
    
    print(f"Original K (distorted):")
    print(f"  fx={K_original[0,0]:.6f}, fy={K_original[1,1]:.6f}")
    print(f"  cx={K_original[0,2]:.6f}, cy={K_original[1,2]:.6f}")
    print()
    print(f"Rectified K (at {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}):")
    print(f"  fx={new_K_rect[0,0]:.6f}, fy={new_K_rect[1,1]:.6f}")
    print(f"  cx={new_K_rect[0,2]:.6f}, cy={new_K_rect[1,2]:.6f}")
    print()
    
    # Step 2: Compute undistortion maps
    print("Step 2: Computing undistortion maps...")
    map1, map2 = cv2.initUndistortRectifyMap(
        K_original,
        dist_coeffs,
        None,  # R = identity (no rotation)
        new_K_rect,
        IMAGE_SIZE,
        cv2.CV_32FC1
    )
    print("Undistortion maps computed.")
    print()
    
    # Step 3: Process all images
    print("Step 3: Processing images...")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Find all images
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(INPUT_DIR).glob(ext))
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {INPUT_DIR}")
        return None
    
    print(f"Found {len(image_files)} images")
    print()
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"WARNING: Could not read {img_file.name}")
            continue
        
        # Check size
        if img.shape[1] != IMAGE_SIZE[0] or img.shape[0] != IMAGE_SIZE[1]:
            print(f"WARNING: {img_file.name} has wrong size {img.shape[1]}x{img.shape[0]}, expected {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
            continue
        
        # Rectify (undistort)
        rectified = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        
        # Save
        output_path = Path(OUTPUT_DIR) / img_file.name
        cv2.imwrite(str(output_path), rectified)
        
        if i % 10 == 0 or i == len(image_files):
            print(f"  Processed {i}/{len(image_files)} images")
    
    print()
    print("=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    
    # Step 4: Save camera parameters
    print("Step 4: Saving camera parameters...")
    yaml_path = Path(OUTPUT_DIR) / "camera_rectified.yaml"
    
    with open(yaml_path, 'w') as f:
        f.write("# General sensor definitions.\n")
        f.write("sensor_type: camera\n")
        f.write("comment: VI-Sensor cam0 (MT9M034) - rectified\n")
        f.write("\n")
        f.write("# Camera specific definitions.\n")
        f.write("rate_hz: 20\n")
        f.write(f"resolution: [{IMAGE_SIZE[0]}, {IMAGE_SIZE[1]}]\n")
        f.write("camera_model: pinhole\n")
        f.write(f"intrinsics: [{new_K_rect[0,0]:.6f}, {new_K_rect[1,1]:.6f}, {new_K_rect[0,2]:.6f}, {new_K_rect[1,2]:.6f}] # fx, fy, cx, cy\n")
        f.write("distortion_model: radial-tangential\n")
        f.write("distortion_coefficients: [0.0, 0.0, 0.0, 0.0]  # rectified, no distortion\n")
    
    print(f"Camera parameters saved to: {yaml_path}")
    print()
    
    # Print summary
    print("SUMMARY:")
    print(f"  Images processed: {len(image_files)}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Resolution: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}")
    print(f"  Intrinsics: fx={new_K_rect[0,0]:.2f}, fy={new_K_rect[1,1]:.2f}, cx={new_K_rect[0,2]:.2f}, cy={new_K_rect[1,2]:.2f}")
    print(f"  Distortion: NONE (rectified)")
    print()
    
    return new_K_rect


if __name__ == "__main__":
    K_rectified = rectify_images()
    
    if K_rectified is not None:
        print("SUCCESS! Your images are ready for COLMAP.")
        print()
        print("Next step: Update your COLMAP integration script with:")
       