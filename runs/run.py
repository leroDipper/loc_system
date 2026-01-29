import torch
import os
from accelerated_modules import vocab_tree_match
from loc_modules.load_gt_params import GroundTruthParams
import cv2
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation
from loc_modules import MapLoader, Localiser
from test.memory import MemoryMonitor
import glob
import yaml

def load_camera_params(yaml_path):
    """Load camera parameters from YAML file."""
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    return {
        'width': params['resolution'][0],
        'height': params['resolution'][1],
        'fx': params['intrinsics'][0],
        'fy': params['intrinsics'][1],
        'cx': params['intrinsics'][2],
        'cy': params['intrinsics'][3]
    }



def load_camera_params_from_colmap(cameras_txt_path):
    """Load camera parameters from COLMAP cameras.txt."""
    with open(cameras_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.strip().split()
            if len(parts) >= 8:
                width = int(parts[2])
                height = int(parts[3])
                
                # PINHOLE: fx fy cx cy
                if parts[1] == 'PINHOLE':
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                elif parts[1] == 'SIMPLE_PINHOLE':
                    f, cx, cy = float(parts[4]), float(parts[5]), float(parts[6])
                    fx = fy = f
                else:
                    fx, fy, cx, cy = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                
                return {
                    'width': width,
                    'height': height,
                    'fx': fx,
                    'fy': fy,
                    'cx': cx,
                    'cy': cy
                }
    
    raise ValueError("No camera found in cameras.txt")

def load_colmap_image_names(images_txt_path):
    """Load image names from COLMAP images.txt in order."""
    colmap_images = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 10:
                colmap_images.append(parts[9])
    return colmap_images

if __name__ == "__main__":
    # Load transformation
    scale, R, t = GroundTruthParams.load_transformation('resources/iphone/colmap_to_apriltag_transform.json')
    CAMERA_PARAMS_PATH = 'resources/iphone/images/camera_rectified.yaml'
    DATASET_PATH = 'resources/iphone'
    IMAGE_DIR = os.path.join(DATASET_PATH, 'images')
    COLMAP_DIR = os.path.join(DATASET_PATH, 'project_files')
    N_TRAIN_IMAGES = 300  # Adjust based on your dataset

    # Load ground truth
    gt_poses = GroundTruthParams.load_iphone_ground_truth(
        colmap_images_txt=os.path.join(COLMAP_DIR, 'images.txt'),
        scale=scale,
        R=R,
        t=t
    )

    xfeat = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        xfeat = xfeat.to(device)
        xfeat.eval()
    except Exception as e:
        print(f"Error loading XFeat model: {e}")
        exit()

    MemoryMonitor.print_memory("After loading XFeat")

    # Load COLMAP image order to determine train/test split
    colmap_images = load_colmap_image_names(os.path.join(COLMAP_DIR, 'images.txt'))
    train_images = set(colmap_images[:N_TRAIN_IMAGES])
    test_images = set(colmap_images[N_TRAIN_IMAGES:])
    
    print(f"COLMAP processed {len(colmap_images)} images total")
    print(f"Train set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")

    # Load existing vocabulary (built with FP32)
    vocabulary = os.path.join(DATASET_PATH, 'vocabularies/vocab_tree_iphone.bin')
    print("Loaded existing vocabulary")

    # Load existing FP32 map (built from train images only)
    data = np.load(os.path.join(DATASET_PATH, 'map_databases/iphone.npz'))
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded FP32 map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

    #camera_params = load_camera_params_from_colmap(os.path.join(COLMAP_DIR, 'cameras.txt'))

    camera_params = load_camera_params(CAMERA_PARAMS_PATH)

    print(f"Loaded camera: {camera_params['width']}x{camera_params['height']}")
    print(f"  fx={camera_params['fx']:.2f}, fy={camera_params['fy']:.2f}")
    print(f"  cx={camera_params['cx']:.2f}, cy={camera_params['cy']:.2f}")

    K_cv = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # Build matcher
    print("\nBuilding vocabulary matcher...")
    t_start = time.time()
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")

    MemoryMonitor.print_memory("After building matcher")

    # ===================================================================
    # CONTINUOUS LOCALISATION TEST
    # ===================================================================
    print("\n" + "="*60)
    print("INT8 QUERIES vs FP32 MAP - IPHONE DATASET")
    print("="*60)
    
    print(f"Map built with {N_TRAIN_IMAGES} images (FP32)")
    print(f"Testing with {len(test_images)} held-out images (INT8)\n")
    
    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}

    # Debug counters
    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    rejected_reproj_error = 0

    # Test only on held-out test images
    test_image_list = sorted(list(test_images))
    
    for i, frame_name in enumerate(test_image_list):
        frame_path = os.path.join(IMAGE_DIR, frame_name)
        
        # Skip if no ground truth available
        if frame_name not in gt_poses:
            skipped_no_gt += 1
            continue
        
        # Read and process frame
        frame = cv2.imread(frame_path)
        if frame is None:
            skipped_no_image += 1
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=200)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()

        # Convert to INT8
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # Match
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80)
        t_match = time.time() - t_start
        
        # Filter unique
        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)
        
        # Build 2D-3D correspondences
        matched_3d = []
        matched_2d = []
        for q_idx, (m_idx, dist) in query_to_map.items():
            matched_3d.append(map_3d_points[m_idx])
            matched_2d.append(keypoints[q_idx])
        
        if len(matched_3d) < 4:  # Need at least 4 points for PnP
            skipped_too_few_matches += 1
            continue
            
        matched_3d = np.array(matched_3d, dtype=np.float32)
        matched_2d = np.array(matched_2d, dtype=np.float32)
        
        # PnP
        t_start = time.time()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_3d,
            matched_2d,
            K_cv,
            dist_coeffs,
            reprojectionError=8.0,
            confidence=0.99
        )
        t_pnp = time.time() - t_start
        
        # Record timings
        timings['extract'].append(t_extract)
        timings['match'].append(t_match)
        timings['pnp'].append(t_pnp)
        
        if not success or len(inliers) < 6:
            pnp_failed += 1
            rejected_low_inliers += 1
            continue
            
        # Check reprojection error of inliers
        inlier_points_3d = matched_3d[inliers.flatten()]
        inlier_points_2d = matched_2d[inliers.flatten()]
        
        projected, _ = cv2.projectPoints(inlier_points_3d, rvec, tvec, K_cv, dist_coeffs)
        reproj_errors = np.linalg.norm(inlier_points_2d - projected.squeeze(), axis=1)
        median_reproj_error = np.median(reproj_errors)
        
        # Reject if median reprojection error too high
        if median_reproj_error > 5.0:
            rejected_reproj_error += 1
            continue
        
        R_cam, _ = cv2.Rodrigues(rvec)
        C_colmap = -R_cam.T @ tvec.flatten()
        C_metres = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
        
        # Accept localisation
        C_gt = gt_poses[frame_name]
        error_metres = np.linalg.norm(C_metres - C_gt)
        
        errors.append(error_metres)
        match_counts.append(len(inliers))
        
        if (i + 1) % 20 == 0:
            print(f"Frame {i+1}/{len(test_image_list)}: "
                  f"Error={error_metres:.3f}m, "
                  f"Matches={len(inliers)}/{len(matched_3d)}")
            MemoryMonitor.print_memory(f"After {i+1} frames")

    print(f"\nDebug Statistics:")
    print(f"  Skipped (no GT): {skipped_no_gt}")
    print(f"  Skipped (no image): {skipped_no_image}")
    print(f"  Skipped (too few matches): {skipped_too_few_matches}")
    print(f"  PnP failed: {pnp_failed}")
    print(f"  Rejected (low inliers): {rejected_low_inliers}")
    print(f"  Rejected (high reproj error): {rejected_reproj_error}")
    
    print("\n" + "="*60)
    print("CONTINUOUS LOCALISATION RESULTS")
    print("="*60)
    print(f"Total test frames: {len(test_image_list)}")
    print(f"Successful localisations: {len(errors)}")
    print(f"Success rate: {len(errors)/len(test_image_list)*100:.1f}%")
    
    if len(errors) > 0:
        print(f"\nLocalisation Accuracy:")
        print(f"  Mean error:   {np.mean(errors):.4f} m ({np.mean(errors)*100:.2f} cm)")
        print(f"  Median error: {np.median(errors):.4f} m ({np.median(errors)*100:.2f} cm)")
        print(f"  Std error:    {np.std(errors):.4f} m")
        print(f"  Max error:    {np.max(errors):.4f} m")
        print(f"  Min error:    {np.min(errors):.4f} m")
        
        print(f"\nMatches per frame:")
        print(f"  Mean: {np.mean(match_counts):.1f}")
        print(f"  Min:  {np.min(match_counts)}")
        print(f"  Max:  {np.max(match_counts)}")
        
        print(f"\nTiming (per frame):")
        print(f"  Feature extraction: {np.mean(timings['extract'])*1000:.2f} ms")
        print(f"  Matching:           {np.mean(timings['match'])*1000:.2f} ms")
        print(f"  PnP:                {np.mean(timings['pnp'])*1000:.2f} ms")
        total_time = np.mean(timings['extract']) + np.mean(timings['match']) + np.mean(timings['pnp'])
        print(f"  Total:              {total_time*1000:.2f} ms")
        print(f"  Average FPS:        {1.0/total_time:.2f}")

        np.savez('results/iphone_train_test_errors.npz', 
            errors=np.array(errors),
            match_counts=np.array(match_counts),
            timings_extract=np.array(timings['extract']),
            timings_match=np.array(timings['match']),
            timings_pnp=np.array(timings['pnp']),
            success_rate=len(errors)/len(test_image_list))
        print("✅ Saved errors to results/iphone_train_test_errors.npz")
    
    MemoryMonitor.print_memory("After continuous localisation")
    
    print("="*60)