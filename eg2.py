#!/usr/bin/env python3
"""
Example: Localisation using OpenCV BFMatcher.
Comparison with main_xfeat.py:
- eg2.py: XFeat/SIFT + OpenCV BFMatcher
- main_xfeat.py: XFeat + Vocabulary-based matching
"""

import torch
import os
import cv2
import numpy as np
import time
import json
from loc_modules.load_gt_params import GroundTruthParams
from loc_modules.matcher import FeatureMatcher
from loc_modules.pose_estimator import PoseEstimator

# Choose feature extractor: 'xfeat' or 'sift'
FEATURE_TYPE = 'xfeat'  # Change to 'sift' to use SIFT features

if __name__ == "__main__":
    print("Loading COLMAP → Ground Truth transformation...")
    scale, R, t = GroundTruthParams.load_transformation('colmap_database/large_map_xfeat/colmap_to_gt_transform.json')

    # Load ground truth positions
    gt_path = 'colmap_database/large_map_xfeat/ground_truth_poses_circ_f8.json'
    gt_poses = GroundTruthParams.load_ground_truth(gt_path)

    # Initialize feature extractor based on choice
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if FEATURE_TYPE == 'xfeat':
        print(f"\nUsing XFeat features on {device}")
        try:
            xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096, trust_repo=True)
            xfeat = xfeat.to(device)
            xfeat.eval()
        except Exception as e:
            print(f"Error loading XFeat model: {e}")
            exit()
        
        # Load XFeat map
        data = np.load('resources/map_databases/colmap_map_train_set.npz')
        map_3d_points = data['xyz_world']
        map_descriptors = data['descriptors']
        print(f"Loaded XFeat map: {len(map_3d_points)} points")
        
    else:  # SIFT
        print("\nUsing SIFT features")
        sift = cv2.SIFT_create()
        
        # Load SIFT map (you'll need a SIFT version of the map)
        # data = np.load('resources/map_databases/colmap_map_train_set_sift.npz')
        # For now, exit if SIFT map doesn't exist
        print("ERROR: SIFT map not available. Please create a SIFT map or use FEATURE_TYPE='xfeat'")
        exit()

    test_dataset_path = 'colmap_database/large_map/large_set_test_640x480'

    # Camera parameters
    camera_params = {
        'width': 640,
        'height': 480,
        'fx': 768.0,
        'fy': 768.0,
        'cx': 320.0,
        'cy': 240.0
    }

    K_cv = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # Initialize matcher and pose estimator
    print("\nInitializing OpenCV BFMatcher...")
    matcher = FeatureMatcher(ratio_threshold=0.75)
    pose_estimator = PoseEstimator(reprojection_error=8.0, confidence=0.99, min_inliers=15)
    pose_estimator.set_map_bounds(map_3d_points)

    # Test image
    frame_name = 'frame_0143.jpg'
    frame_path = os.path.join(test_dataset_path, frame_name)
    
    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract features
    t_start = time.time()
    if FEATURE_TYPE == 'xfeat':
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=100)
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
    else:  # SIFT
        keypoints_cv, descriptors = sift.detectAndCompute(frame_gray, None)
        keypoints = np.array([kp.pt for kp in keypoints_cv], dtype=np.float32)
        descriptors = descriptors.astype(np.float32)
    
    t_extract = time.time() - t_start
    print(f"Extracted {len(keypoints)} features in {t_extract:.3f}s")

    # Match
    t_start = time.time()
    matched_map_indices, matched_2d_points = matcher.match(
        map_descriptors,
        descriptors,
        keypoints
    )
    t_match = time.time() - t_start
    
    if matched_map_indices is None:
        print("Matching failed - not enough matches")
        exit()
    
    print(f"Matched {len(matched_map_indices)} features in {t_match:.3f}s")

    # Get corresponding 3D points
    matched_3d_points = map_3d_points[matched_map_indices]

    # PnP
    t_start = time.time()
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        matched_3d_points,
        matched_2d_points,
        K_cv,
        dist_coeffs,
        reprojectionError=8.0,
        confidence=0.99
    )
    t_pnp = time.time() - t_start

    if success:
        R_cam, _ = cv2.Rodrigues(rvec)
        C_colmap = -R_cam.T @ tvec.flatten()
        
        # Transform to meters
        C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
        
        # Get ground truth
        C_gt = gt_poses[frame_name]
        
        # Compute error
        error_meters = np.linalg.norm(C_meters - C_gt)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Estimated (meters) {FEATURE_TYPE.upper()} + BFMatcher: {C_meters}")
        print(f"Ground truth:        {C_gt}")
        print(f"\nError: {error_meters:.4f} meters ({error_meters*100:.2f} cm)")
        print(f"Inliers: {len(inliers)}/{len(matched_3d_points)}")

        print(f"\n{'='*60}")
        print("TIMING")
        print(f"{'='*60}")
        print(f"Feature extraction: {t_extract:.3f}s")
        print(f"Matching:           {t_match:.3f}s")
        print(f"PnP:                {t_pnp:.3f}s")
        print(f"Total per frame:    {t_extract + t_match + t_pnp:.3f}s")
        print(f"FPS:                {1.0/(t_extract + t_match + t_pnp):.2f}")
    else:
        print("PnP failed to find a solution.")