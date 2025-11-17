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

def pose_from_pnp(rvec, tvec, scale, R, t):
    """Convert PnP result to camera center in ground truth coordinates."""
    R_cam, _ = cv2.Rodrigues(rvec)
    C_colmap = -R_cam.T @ tvec.flatten()
    C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
    return C_meters, R_cam, tvec

if __name__ == "__main__":
    scale, R, t = GroundTruthParams.load_transformation('resources/tum_fr2/colmap_to_gt_transform.json')
    CAMERA_PARAMS_PATH = 'resources/tum_fr2/camera_params.yaml'
    TUM_DATASET_PATH = 'resources/tum_fr2'

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

    test_dataset_path = 'resources/tum_fr2/images'

    # Load existing vocabulary
    vocabulary = 'resources/tum_fr2/vocabularies/vocab_tree.bin'
    print("Loaded existing vocabulary")

    # Load colmap map
    data = np.load('resources/tum_fr2/map_databases/tumfr2_map_train.npz')
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
    # CONTINUOUS LOCALIZATION TEST WITH TEMPORAL TRACKING
    # ===================================================================
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION TEST - TUM FR2 (WITH TEMPORAL TRACKING)")
    print("="*60)
    
    # Get all test frames (skip first 2000 used for map building)
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[2000:]  # Use images after first 2000
    
    print(f"Map built with first 2000 images")
    print(f"Testing with {len(test_frames)} remaining images\n")
    
    errors = []
    match_counts = []
    localization_modes = []  # Track which method was used
    timings = {'extract': [], 'match': [], 'pnp': [], 'tracking': []}
    
    # DEBUG COUNTERS
    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    temporal_success = 0
    temporal_failed = 0
    global_success = 0
    
    # TEMPORAL TRACKING STATE
    previous_frame_gray = None
    tracking_map_points = None  # 3D points being tracked
    tracking_keypoints = None   # Their 2D locations in previous frame
    frames_since_global = 0
    MAX_TRACKING_FRAMES = 1    # Force global relocalization after this many tracking frames
    
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
        
        current_pose = None
        mode = "NONE"
        num_matches = 0
        t_total_start = time.time()
        
        # ============================================================
        # STEP 1: TRY GLOBAL MAP LOCALIZATION (PRIMARY METHOD)
        # ============================================================
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=200)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # Match
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.9)
        t_match = time.time() - t_start
        
        # Filter unique
        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)
        
        # Build 2D-3D correspondences
        matched_3d = []
        matched_2d = []
        matched_query_idx = []
        for q_idx, (m_idx, dist) in query_to_map.items():
            matched_3d.append(map_3d_points[m_idx])
            matched_2d.append(keypoints[q_idx])
            matched_query_idx.append(q_idx)
        
        global_localization_attempted = len(matched_3d) >= 4
        
        if global_localization_attempted:
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
            timings['pnp'].append(t_pnp)
            
            if success and len(inliers) >= 6:
                # Global localization succeeded!
                C_meters, R_cam, tvec_final = pose_from_pnp(rvec, tvec, scale, R, t)
                current_pose = C_meters
                num_matches = len(inliers)
                mode = "GLOBAL"
                global_success += 1
                frames_since_global = 0
                
                # Save tracking state for next frame
                inlier_indices = inliers.flatten()
                tracking_map_points = matched_3d[inlier_indices]
                tracking_keypoints = matched_2d[inlier_indices]
                previous_frame_gray = frame_gray.copy()
            else:
                if not success:
                    pnp_failed += 1
                else:
                    rejected_low_inliers += 1
        else:
            skipped_too_few_matches += 1
        
        # ============================================================
        # STEP 2: IF GLOBAL FAILED, TRY TEMPORAL TRACKING (BACKUP)
        # ============================================================
        if current_pose is None and tracking_keypoints is not None and frames_since_global < MAX_TRACKING_FRAMES:
            t_track_start = time.time()

            print(f"  Frame {i}: Attempting temporal (have {len(tracking_keypoints)} points to track)")



            
            # Track features from previous frame to current frame
            tracked_keypoints, status, err = cv2.calcOpticalFlowPyrLK(
                previous_frame_gray,
                frame_gray,
                tracking_keypoints.astype(np.float32),
                None,
                winSize=(21, 21),
                maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            
            t_tracking = time.time() - t_track_start
            timings['tracking'].append(t_tracking)
            
            # Keep only successfully tracked points
            if status is not None:
                valid = status.flatten() == 1
                print(f"    Tracked {valid.sum()}/{len(tracking_keypoints)} points successfully")
                valid_3d = tracking_map_points[valid]
                valid_2d = tracked_keypoints[valid]


                
                if len(valid_3d) >= 6:
                    print(f"    Attempting PnP with {len(valid_3d)} tracked points...")
                    # Solve PnP with tracked correspondences
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        valid_3d,
                        valid_2d,
                        K_cv,
                        dist_coeffs,
                        reprojectionError=8.0,
                        confidence=0.99
                    )
                    
                    if success and len(inliers) >= 6:
                        inlier_3d = valid_3d[inliers.flatten()]
                        inlier_2d = valid_2d[inliers.flatten()]
                        projected, _ = cv2.projectPoints(inlier_3d, rvec, tvec, K_cv, dist_coeffs)
                        reproj_errors = np.linalg.norm(inlier_2d - projected.squeeze(), axis=1)
                        median_reproj = np.median(reproj_errors)
                        print(f"      PnP success: {len(inliers)} inliers, median_reproj={median_reproj:.2f}px")

                        inlier_ratio = len(inliers) / len(valid_3d)
                        
                        if median_reproj < 3.0 and inlier_ratio > 0.7:

                            # Temporal tracking succeeded!
                            C_meters, R_cam, tvec_final = pose_from_pnp(rvec, tvec, scale, R, t)
                            current_pose = C_meters
                            num_matches = len(inliers)
                            mode = "TEMPORAL"
                            temporal_success += 1
                            frames_since_global += 1
                            
                            # Update tracking state (keep only inliers)
                            inlier_indices = inliers.flatten()
                            tracking_map_points = valid_3d[inlier_indices]
                            tracking_keypoints = valid_2d[inlier_indices]
                            previous_frame_gray = frame_gray.copy()
                    else:
                        temporal_failed += 1
                        tracking_keypoints = None  # Lost tracking
                else:
                    temporal_failed += 1
                    tracking_keypoints = None  # Not enough tracked points
            else:
                temporal_failed += 1
                tracking_keypoints = None
        
        # ============================================================
        # STEP 3: RECORD RESULTS
        # ============================================================
        timings['extract'].append(t_extract)
        timings['match'].append(t_match)
        
        if current_pose is not None:
            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(current_pose - C_gt)
            
            errors.append(error_meters)
            match_counts.append(num_matches)
            localization_modes.append(mode)
            
            if (i + 1) % 20 == 0:
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Mode={mode:8s}, Error={error_meters:.3f}m, Matches={num_matches}")
        else:
            # Lost tracking
            tracking_keypoints = None
            frames_since_global = 0
            
        if (i + 1) % 100 == 0:
            MemoryMonitor.print_memory(f"After {i+1} frames")
    
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION RESULTS")
    print("="*60)
    print(f"Total frames attempted: {len(test_frames)}")
    print(f"Successful localizations: {len(errors)}")
    print(f"Success rate: {len(errors)/len(test_frames)*100:.1f}%")
    
    print(f"\n{'='*60}")
    print("LOCALIZATION MODE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Global map localizations:   {global_success} ({global_success/len(test_frames)*100:.1f}%)")
    print(f"Temporal tracking:          {temporal_success} ({temporal_success/len(test_frames)*100:.1f}%)")
    print(f"Total successful:           {len(errors)} ({len(errors)/len(test_frames)*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print("FAILURE BREAKDOWN")
    print(f"{'='*60}")
    print(f"Skipped - no ground truth:     {skipped_no_gt}")
    print(f"Skipped - image failed to load: {skipped_no_image}")
    print(f"Skipped - too few matches (<4): {skipped_too_few_matches}")
    print(f"PnP RANSAC failed:              {pnp_failed}")
    print(f"Rejected - low inliers (<6):    {rejected_low_inliers}")
    print(f"Temporal tracking failed:       {temporal_failed}")
    
    if len(errors) > 0:
        print(f"\n{'='*60}")
        print("LOCALIZATION ACCURACY")
        print(f"{'='*60}")
        print(f"Mean error:   {np.mean(errors):.4f} m ({np.mean(errors)*100:.2f} cm)")
        print(f"Median error: {np.median(errors):.4f} m ({np.median(errors)*100:.2f} cm)")
        print(f"Std error:    {np.std(errors):.4f} m")
        print(f"Max error:    {np.max(errors):.4f} m")
        print(f"Min error:    {np.min(errors):.4f} m")
        
        threshold_20cm = sum(e < 0.20 for e in errors)
        print(f"\nFrames within 20cm: {threshold_20cm}/{len(errors)} ({threshold_20cm/len(errors)*100:.1f}%)")
        
        # Accuracy by mode
        global_errors = [e for e, m in zip(errors, localization_modes) if m == "GLOBAL"]
        temporal_errors = [e for e, m in zip(errors, localization_modes) if m == "TEMPORAL"]
        
        if len(global_errors) > 0:
            print(f"\nGlobal localization accuracy:")
            print(f"  Mean: {np.mean(global_errors):.4f} m ({np.mean(global_errors)*100:.2f} cm)")
            print(f"  Median: {np.median(global_errors):.4f} m")
        
        if len(temporal_errors) > 0:
            print(f"\nTemporal tracking accuracy:")
            print(f"  Mean: {np.mean(temporal_errors):.4f} m ({np.mean(temporal_errors)*100:.2f} cm)")
            print(f"  Median: {np.median(temporal_errors):.4f} m")
        
        print(f"\nMatches per frame:")
        print(f"  Mean: {np.mean(match_counts):.1f}")
        print(f"  Min:  {np.min(match_counts)}")
        print(f"  Max:  {np.max(match_counts)}")
        
        print(f"\nTiming (per frame):")
        print(f"  Feature extraction: {np.mean(timings['extract'])*1000:.2f} ms")
        print(f"  Matching:           {np.mean(timings['match'])*1000:.2f} ms")
        if len(timings['pnp']) > 0:
            print(f"  PnP (global):       {np.mean(timings['pnp'])*1000:.2f} ms")
        if len(timings['tracking']) > 0:
            print(f"  Optical flow:       {np.mean(timings['tracking'])*1000:.2f} ms")
        total_time = np.mean(timings['extract']) + np.mean(timings['match']) + \
                     (np.mean(timings['pnp']) if len(timings['pnp']) > 0 else 0)
        print(f"  Total (avg):        {total_time*1000:.2f} ms")
        print(f"  Average FPS:        {1.0/total_time:.2f}")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)