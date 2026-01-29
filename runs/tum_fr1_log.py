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
from loc_modules.featureLife import FeatureLifecycleLogger

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
    scale, R, t = GroundTruthParams.load_transformation('resources/tum_fr1/colmap_to_gt_transform.json')
    CAMERA_PARAMS_PATH = 'resources/tum_fr1/camera_params.yaml'
    TUM_DATASET_PATH = 'resources/tum_fr1'

    gt_poses = GroundTruthParams.load_tum_ground_truth(
        gt_file_path=os.path.join(TUM_DATASET_PATH, 'groundtruth.txt'),
        rgb_file_path=os.path.join(TUM_DATASET_PATH, 'rgb.txt')
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

    test_dataset_path = 'resources/tum_fr1/images'

    # Load existing vocabulary
    vocabulary = 'resources/tum_fr1/vocabularies/vocab_tree.bin'
    print("Loaded existing vocabulary")

    # Load colmap map
    data = np.load('resources/tum_fr1/map_databases/tumfr1_map_train.npz')
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

    # Initialize feature lifecycle logger
    logger = FeatureLifecycleLogger()
    print("Feature lifecycle logger initialized\n")

    # ===================================================================
    # CONTINUOUS LOCALIZATION TEST
    # ===================================================================
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION TEST - TUM FR1")
    print("="*60)
    
    # Get all test frames (skip first 500 used for map building)
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[0:]  # Use images after first 500
    
    print(f"Map built with first 500 images")
    print(f"Testing with {len(test_frames)} remaining images\n")
    
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
        
        # Start logging this frame
        logger.start_frame(i, frame_name)
        
        # Skip if no ground truth available
        if frame_name not in gt_poses:
            continue
        
        # Read and process frame
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=100)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors_fp32 = features['descriptors'].cpu().numpy()
        scores = features['scores'].cpu().numpy()
        
        # LOG: Stage 1 - Extraction
        frame_features = logger.log_extraction(keypoints, descriptors_fp32, scores)
        
        # Quantize descriptors
        descriptors = np.clip((descriptors_fp32 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # LOG: Stage 2 - Quantization
        logger.log_quantization(frame_features, descriptors)
        
        # Match
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80, k_nearest_words=4)
        t_match = time.time() - t_start
        
        # LOG: Stage 4 - Matching (placeholder for vocab assignment - Stage 3)
        # Note: We'll do vocab logging later when we expose it from matcher
        word_ids = np.zeros(len(descriptors), dtype=np.int32)  # Placeholder
        distances_to_center = np.zeros(len(descriptors))
        logger.log_vocab_assignment(frame_features, word_ids, distances_to_center)
        
        # Build proper match statistics for logging
        # Group matches by query index to find best and second-best
        query_match_info = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_match_info:
                query_match_info[q_idx] = []
            query_match_info[q_idx].append((m_idx, dist))
        
        # Sort each query's matches by distance
        for q_idx in query_match_info:
            query_match_info[q_idx].sort(key=lambda x: x[1])
        
        # Count candidates per query (for uncertainty measure)
        n_candidates_per_query = {q_idx: len(matches) for q_idx, matches in query_match_info.items()}
        
        # LOG: Matching stage
        logger.log_matching(frame_features, query_idx, map_idx, distances, n_candidates_per_query)
        
        # Filter unique
        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)
        
        # Build 2D-3D correspondences
        matched_3d = []
        matched_2d = []
        matched_query_indices = []  # Track which features were matched
        for q_idx, (m_idx, dist) in query_to_map.items():
            matched_3d.append(map_3d_points[m_idx])
            matched_2d.append(keypoints[q_idx])
            matched_query_indices.append(q_idx)
        
        if len(matched_3d) < 4:  # Need at least 4 points for PnP
            # LOG: Failed - too few matches
            logger.log_ransac(frame_features, np.array([]), np.array([]), None)
            logger.log_final_pose(frame_features, np.array([]), np.array([]), False)
            continue
            
        matched_3d = np.array(matched_3d, dtype=np.float32)
        matched_2d = np.array(matched_2d, dtype=np.float32)
        matched_query_indices = np.array(matched_query_indices, dtype=np.int32)
        
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
        
        # Compute reprojection errors for all matched points
        if success and len(inliers) > 0:
            projected, _ = cv2.projectPoints(matched_3d, rvec, tvec, K_cv, dist_coeffs)
            reproj_errors = np.linalg.norm(matched_2d - projected.squeeze(), axis=1)
            
            # Map back to original feature indices
            inlier_feature_indices = matched_query_indices[inliers.flatten()]
            
            # Create array of reprojection errors aligned with matched_query_indices
            reproj_errors_aligned = reproj_errors
            
            # LOG: Stage 5 - RANSAC
            logger.log_ransac(
                frame_features,
                inlier_feature_indices,
                reproj_errors_aligned,
                all_tested_indices=matched_query_indices
            )
        else:
            # RANSAC failed
            logger.log_ransac(frame_features, np.array([]), np.array([]), matched_query_indices)
        
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
                # LOG: Rejected
                logger.log_final_pose(frame_features, inlier_feature_indices, reproj_errors, False)
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
            
            # LOG: Stage 6 - Final pose contribution
            logger.log_final_pose(frame_features, inlier_feature_indices, reproj_errors, True)
            
            if (i + 1) % 20 == 0:
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}")
                MemoryMonitor.print_memory(f"After {i+1} frames")
        else:
            rejected_low_inliers += 1
            # LOG: Failed - too few inliers
            if success:
                logger.log_final_pose(frame_features, inlier_feature_indices, reproj_errors, False)
            else:
                logger.log_final_pose(frame_features, np.array([]), np.array([]), False)

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

        np.savez('results/fr1_fp32_errors.npz', 
            errors=np.array(errors),
             match_counts=np.array(match_counts),
             timings_extract=np.array(timings['extract']),
             timings_match=np.array(timings['match']),
             timings_pnp=np.array(timings['pnp']),
             success_rate=len(errors)/len(test_frames))
        print("✓ Saved errors to results/fr1_fp32_errors.npz")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)
    
    # ===================================================================
    # FEATURE LIFECYCLE ANALYSIS
    # ===================================================================
    
    print("\n" + "="*60)
    print("FEATURE LIFECYCLE ANALYSIS")
    print("="*60)
    
    # Save raw lifecycle data
    logger.save('results/feature_lifecycles.csv')
    
    # Print summary
    logger.print_summary()
    
    # Export to dataframe for analysis
    df = logger.export_to_dataframe()
    
    # Analyze survivors vs failures
    survivors = df[df['survived'] == True]
    failures = df[df['survived'] == False]
    
    if len(survivors) > 0 and len(failures) > 0:
        print(f"\n{'='*60}")
        print(f"SURVIVOR VS FAILURE ANALYSIS")
        print(f"{'='*60}")
        
        measures = [
            ('detector_score', 'Detector Score'),
            ('quantization_error', 'Quantization Error'),
            ('distance_to_word_center', 'Distance to Vocab Center'),
            ('n_candidates', 'Number of Candidates'),
            ('ratio_test_value', 'Ratio Test Value'),
            ('reprojection_error', 'Reprojection Error'),
        ]
        
        for measure, label in measures:
            if measure in df.columns and df[measure].notna().any():
                surv_mean = survivors[measure].mean()
                fail_mean = failures[measure].mean()
                
                if not np.isnan(surv_mean) and not np.isnan(fail_mean):
                    diff = surv_mean - fail_mean
                    percent_diff = (diff / (fail_mean + 1e-8)) * 100
                    
                    print(f"\n{label}:")
                    print(f"  Survivors: {surv_mean:.4f}")
                    print(f"  Failures:  {fail_mean:.4f}")
                    print(f"  Difference: {diff:+.4f} ({percent_diff:+.1f}%)")
        
        # Save survivors for further analysis
        survivors.to_csv('results/survivors_only.csv', index=False)
        print(f"\n✓ Saved {len(survivors)} survivor features to results/survivors_only.csv")
    
    print("="*60)