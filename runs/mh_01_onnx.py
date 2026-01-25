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
import yaml
from test.memory import MemoryMonitor
import glob
from loc_modules.onnx_feat import onnx_extractor
import onnxruntime as ort

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

if __name__ == "__main__":
    scale, R, t = GroundTruthParams.load_transformation('resources/mh_01/colmap_files/colmap_to_gt_transform_int8.json')
    CAMERA_PARAMS_PATH = 'resources/mh_01/images/camera_rectified.yaml'
    EUROC_DATASET_PATH = 'resources/mh_01'
    

    gt_poses = GroundTruthParams.load_euroc_ground_truth_by_image(
        gt_csv_path=os.path.join(EUROC_DATASET_PATH, 'data.csv'),
        image_dir=os.path.join(EUROC_DATASET_PATH, 'images')
    )


    # Load INT8 ONNX model instead of PyTorch
    print("Loading INT8 ONNX model...")
    model='models/xfeat_752x480_int8.onnx'
    session = ort.InferenceSession(model, providers=['CPUExecutionProvider'])
    print("✓ model loaded")

    MemoryMonitor.print_memory("After loading INT8 ONNX model")



    test_dataset_path = 'resources/mh_01/images_small'

    # Load existing vocabulary
    vocabulary = 'resources/mh_01/vocabularies/vocab_tree_mh_01_quant.bin'
    print("Loaded existing vocabulary")

    # Load colmap map
    data = np.load('resources/mh_01/map_databases/mh_01_quant.npz')

    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

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
    # CONTINUOUS LOCALIZATION TEST
    # ===================================================================
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION TEST - Mh")
    print("="*60)
    
    # Load registered images from COLMAP reconstruction
    colmap_images_path = 'resources/mh_01/colmap_files/images.txt'
    registered_images = []
    with open(colmap_images_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 10:
                registered_images.append(parts[9])
    
    # Split into train/test based on registered images only
    n_train = 0
    test_image_names = set(registered_images[n_train:])
    
    # Get test frames - only those that were registered by COLMAP
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = [f for f in all_frames if os.path.basename(f) in test_image_names]
    
    print(f"COLMAP registered: {len(registered_images)} images")
    print(f"Train images: {n_train}")
    print(f"Test images: {len(test_frames)}\n")
    
    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}

     # DEBUG COUNTERS
    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    rejected_reproj_error = 0


    
    for i, frame_path in enumerate(test_frames):
        frame_name = os.path.basename(frame_path)
        
        # Skip if no ground truth available
        if frame_name not in gt_poses:
            continue
        
        # Read and process frame
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        
        keypoints, descriptors, t_extract = onnx_extractor(session, frame_gray, top_k=200)
        

         # Convert descriptors to uint8 for matching
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
        
        if success and len(inliers) >= 6:
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
            C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
            
            # Accept localization
            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            errors.append(error_meters)
            match_counts.append(len(inliers))
            previous_C = C_meters
            
            if (i + 1) % 20 == 0:
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}")
                MemoryMonitor.print_memory(f"After {i+1} frames")
        else:
            rejected_low_inliers += 1

    print(f"Total test frames available: {len(test_frames)}")
    print(f"Frames with GT: {sum(1 for f in test_frames if os.path.basename(f) in gt_poses)}")
    
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION RESULTS")
    print("="*60)
    print(f"Total frames attempted: {len(test_frames)}")
    print(f"Successful localizations: {len(errors)}")
    print(f"Success rate: {len(errors)/len(test_frames)*100:.1f}%")
    
    if len(errors) > 0:
        print(f"\nLocalization Accuracy:")
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

        # np.savez('results/fr1_fp32_errors.npz', 
        #     errors=np.array(errors),
        #      match_counts=np.array(match_counts),
        #      timings_extract=np.array(timings['extract']),
        #      timings_match=np.array(timings['match']),
        #      timings_pnp=np.array(timings['pnp']),
        #      success_rate=len(errors)/len(test_frames))
        # print("✓ Saved errors to results/fr1_fp32_errors.npz")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)