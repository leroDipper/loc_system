#!/usr/bin/env python3
"""
Diagnostic: Check if rectification produced correct intrinsics
"""

import cv2
import numpy as np
import yaml

# Original parameters
K_original = np.array([
    [458.654, 0, 367.215],
    [0, 457.296, 248.375],
    [0, 0, 1]
], dtype=np.float64)

dist_coeffs = np.array([
    -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05
], dtype=np.float64)

ORIGINAL_SIZE = (752, 480)
TARGET_SIZE = (640, 480)

print("=" * 60)
print("RECTIFICATION DIAGNOSTIC")
print("=" * 60)

# Step 1: What does getOptimalNewCameraMatrix give us?
print("\n1. Computing rectified intrinsics at ORIGINAL size (752x480):")
new_K_rect, roi = cv2.getOptimalNewCameraMatrix(
    K_original, 
    dist_coeffs, 
    ORIGINAL_SIZE, 
    alpha=0,
    newImgSize=ORIGINAL_SIZE
)

print(f"   Rectified K at 752x480:")
print(f"   fx = {new_K_rect[0,0]:.6f}")
print(f"   fy = {new_K_rect[1,1]:.6f}")
print(f"   cx = {new_K_rect[0,2]:.6f}")
print(f"   cy = {new_K_rect[1,2]:.6f}")
print(f"   fx/fy ratio = {new_K_rect[0,0]/new_K_rect[1,1]:.6f}")

# Step 2: Scale to target size
print(f"\n2. Scaling to TARGET size (640x480):")
scale_x = TARGET_SIZE[0] / ORIGINAL_SIZE[0]
scale_y = TARGET_SIZE[1] / ORIGINAL_SIZE[1]

print(f"   scale_x = {TARGET_SIZE[0]}/{ORIGINAL_SIZE[0]} = {scale_x:.6f}")
print(f"   scale_y = {TARGET_SIZE[1]}/{ORIGINAL_SIZE[1]} = {scale_y:.6f}")

K_final = new_K_rect.copy()
K_final[0, 0] *= scale_x  # fx
K_final[1, 1] *= scale_y  # fy  
K_final[0, 2] *= scale_x  # cx
K_final[1, 2] *= scale_y  # cy

print(f"\n   Final K at 640x480:")
print(f"   fx = {K_final[0,0]:.6f}")
print(f"   fy = {K_final[1,1]:.6f}")
print(f"   cx = {K_final[0,2]:.6f}")
print(f"   cy = {K_final[1,2]:.6f}")
print(f"   fx/fy ratio = {K_final[0,0]/K_final[1,1]:.6f}")

# Step 3: Load what's in the YAML
print(f"\n3. What's in camera_rectified.yaml:")
yaml_path = "/home/leroy/masters_local/masfiles/test_images_mh/images/data_640x480/camera_rectified.yaml"
try:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    fx, fy, cx, cy = data['intrinsics']
    print(f"   fx = {fx:.6f}")
    print(f"   fy = {fy:.6f}")
    print(f"   cx = {cx:.6f}")
    print(f"   cy = {cy:.6f}")
    print(f"   fx/fy ratio = {fx/fy:.6f}")
    
    # Compare
    print(f"\n4. COMPARISON:")
    print(f"   Expected fx: {K_final[0,0]:.6f}")
    print(f"   Actual fx:   {fx:.6f}")
    print(f"   Difference:  {abs(K_final[0,0] - fx):.6f}")
    print()
    print(f"   Expected fy: {K_final[1,1]:.6f}")
    print(f"   Actual fy:   {fy:.6f}")
    print(f"   Difference:  {abs(K_final[1,1] - fy):.6f}")
    
    if abs(K_final[0,0] - fx) > 1.0 or abs(K_final[1,1] - fy) > 1.0:
        print("\n   ⚠️  MISMATCH DETECTED!")
        print("   Your YAML has incorrect parameters!")
    else:
        print("\n   ✓ Parameters match!")
        
except FileNotFoundError:
    print(f"   File not found: {yaml_path}")

print("\n" + "=" * 60)