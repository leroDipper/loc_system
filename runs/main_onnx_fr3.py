"""
main_tumfr1_int8.py

Test INT8 ONNX model with existing FP32 map.
"""

import onnxruntime as ort
import os
from accelerated_modules import vocab_tree_match
from loc_modules.load_gt_params import GroundTruthParams
import cv2
import numpy as np
import time
import yaml
from test.memory import MemoryMonitor
import glob

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
    scale, R, t = GroundTruthParams.load_transformation('resources/tum_fr3/colmap_to_gt_transform.json')
    CAMERA_PARAMS_PATH = 'resources/tum_fr3/camera_params.yaml'
    TUM_DATASET_PATH = 'resources/tum_fr3'

    gt_poses = GroundTruthParams.load_tum_ground_truth(
        gt_file_path=os.path.join(TUM_DATASET_PATH, 'groundtruth.txt'),
        rgb_file_path=os.path.join(TUM_DATASET_PATH, 'rgb.txt')
    )

    # Load INT8 ONNX model instead of PyTorch
    print("Loading INT8 ONNX model...")
    int8_model_path = 'models/xfeat_640x480_int8.onnx'
    session = ort.InferenceSession(int8_model_path, providers=['CPUExecutionProvider'])
    print("✓ INT8 model loaded")

    MemoryMonitor.print_memory("After loading INT8 ONNX model")

    test_dataset_path = 'resources/tum_fr3/images'

    # Load existing vocabulary (built with FP32)
    vocabulary = 'resources/tum_fr1/vocabularies/vocab_tree.bin'
    print("Loaded existing vocabulary")

    # Load existing FP32 map
    data = np.load('resources/tum_fr3/map_databases/tumfr3_map_train.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded FP32 map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

    camera_params = load_camera_params(CAMERA_PARAMS_PATH)
    print(f"Loaded camera: {camera_params['width']}x{camera_params['height']}")
    print(f"  fx={camera_params['fx']:.2f}, fy={camera_params['fy']:.2f}")
    print(f"  cx={camera_params['cx']:.2f}, cy={camera_params['cy']:.2f}")

    K_cv = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # Build matcher with FP32 map descriptors
    print("\nBuilding vocabulary matcher...")
    t_start = time.time()
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")

    MemoryMonitor.print_memory("After building matcher")

    # ===================================================================
    # CONTINUOUS LOCALIZATION TEST - INT8 queries vs FP32 map
    # ===================================================================
    print("\n" + "="*60)
    print("INT8 QUERIES vs FP32 MAP - TUM FR3")
    print("="*60)
    
    # Get all test frames (skip first 500 used for map building)
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[1500:]  # Use images after first 1500
    
    print(f"Map built with first 500 images (FP32)")
    print(f"Testing with {len(test_frames)} remaining images (INT8)\n")
    
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
            skipped_no_gt += 1
            continue
        
        # Read and process frame
        frame = cv2.imread(frame_path)
        if frame is None:
            skipped_no_image += 1
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Preprocess for ONNX
        t_start = time.time()
        frame_input = frame_gray.astype(np.float32)
        frame_input = np.expand_dims(frame_input, axis=0)
        frame_input = np.expand_dims(frame_input, axis=0)
        
        # Extract features using INT8 ONNX
        feats, keypoints_logits, heatmap = session.run(None, {'input': frame_input})
        t_extract = time.time() - t_start
        
        # Process outputs to get sparse features (top 100)
        B, C, H, W = feats.shape
        
        # Get heatmap scores and select top-k
        heat_flat = heatmap[0, 0].flatten()
        top_k = 200
        
        if len(heat_flat) > top_k:
            top_indices = np.argpartition(heat_flat, -top_k)[-top_k:]
        else:
            top_indices = np.arange(len(heat_flat))
        
        # Get keypoint positions
        y_coords = np.repeat(np.arange(H), W)
        x_coords = np.tile(np.arange(W), H)
        kpts_x = x_coords[top_indices] * 8
        kpts_y = y_coords[top_indices] * 8
        keypoints = np.stack([kpts_x, kpts_y], axis=1).astype(np.float32)
        
        # Get descriptors
        feats_flat = feats[0].reshape(64, -1).T
        descriptors = feats_flat[top_indices]
        
        # L2 normalize
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        descriptors = descriptors / (norms + 1e-8)
        
        # Convert to uint8 (to match map format)
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # Match INT8 descriptors against FP32 map
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.85)
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
            reprojectionError=12.0,
            confidence=0.95
        )
        t_pnp = time.time() - t_start
        
        # Record timings
        timings['extract'].append(t_extract)
        timings['match'].append(t_match)
        timings['pnp'].append(t_pnp)

        if not success:
            pnp_failed += 1
            continue
        
        if success and len(inliers) >= 6:
            # Check reprojection error of inliers
            inlier_points_3d = matched_3d[inliers.flatten()]
            inlier_points_2d = matched_2d[inliers.flatten()]
            
            projected, _ = cv2.projectPoints(
                inlier_points_3d,
                rvec,
                tvec,
                K_cv,
                dist_coeffs
            )
            reproj_errors = np.linalg.norm(inlier_points_2d - projected.squeeze(), axis=1)
            median_reproj_error = np.median(reproj_errors)


            # Reject if median reprojection error too high
            if median_reproj_error > 5.0:
                rejected_reproj_error += 1
                continue





            R_cam, _ = cv2.Rodrigues(rvec)
            C_colmap = -R_cam.T @ tvec.flatten()
            C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)

            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            errors.append(error_meters)
            match_counts.append(len(inliers))
            
            if (i + 1) % 50 == 0:
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}")
                MemoryMonitor.print_memory(f"After {i+1} frames")
        else:
            rejected_low_inliers += 1
    
    print("\n" + "="*60)
    print("INT8 vs FP32 MAP RESULTS")
    print("="*60)
    print(f"Total frames attempted: {len(test_frames)}")
    print(f"Successful localizations: {len(errors)}")
    print(f"Success rate: {len(errors)/len(test_frames)*100:.1f}%")

    print(f"\n{'='*60}")
    print("FAILURE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Skipped - no ground truth:      {skipped_no_gt}")
    print(f"Skipped - image failed to load: {skipped_no_image}")
    print(f"Skipped - too few matches (<4): {skipped_too_few_matches}")
    print(f"PnP RANSAC failed:              {pnp_failed}")
    print(f"Rejected - low inliers (<6):    {rejected_low_inliers}")
    print(f"Rejected - high reproj. error:  {rejected_reproj_error}")
    total_failures = skipped_no_gt + skipped_no_image + skipped_too_few_matches + pnp_failed + rejected_low_inliers + rejected_reproj_error
    print(f"Total failures:                 {total_failures}")
    print(f"Sanity check (should = {len(test_frames)}): {len(errors) + total_failures}")
    
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

        np.savez('results/fr3_int8_errors.npz',
             errors=np.array(errors),
             match_counts=np.array(match_counts),
             timings_extract=np.array(timings['extract']),
             timings_match=np.array(timings['match']),
             timings_pnp=np.array(timings['pnp']),
             success_rate=len(errors)/len(test_frames))
        print("✓ Saved errors to results/fr3_int8_errors.npz")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)