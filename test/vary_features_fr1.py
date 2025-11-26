"""
vary_features_fr1.py

Experiment: Vary number of features (N) to characterize accuracy-speed trade-off.
Platform: Development machine (CPU only)
Dataset: TUM FR1 (test frames 500+)
Goal: Inform feature count selection for embedded deployment
"""

import torch
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
    # Configuration
    scale, R, t = GroundTruthParams.load_transformation('resources/tum_fr1/colmap_to_gt_transform.json')
    CAMERA_PARAMS_PATH = 'resources/tum_fr1/camera_params.yaml'
    TUM_DATASET_PATH = 'resources/tum_fr1'

    # Load ground truth
    gt_poses = GroundTruthParams.load_tum_ground_truth(
        gt_file_path=os.path.join(TUM_DATASET_PATH, 'groundtruth.txt'),
        rgb_file_path=os.path.join(TUM_DATASET_PATH, 'rgb.txt')
    )

    # Load XFeat model - FORCE CPU
    print("="*60)
    print("FEATURE COUNT VARIATION EXPERIMENT - TUM FR1")
    print("="*60)
    print("Platform: CPU only (development machine)")
    print("Loading XFeat model...")
    
    device = "cpu"  # FORCE CPU
    
    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        xfeat = xfeat.to(device)
        xfeat.eval()
        print(f"✓ XFeat loaded on {device}")
    except Exception as e:
        print(f"Error loading XFeat model: {e}")
        exit()

    MemoryMonitor.print_memory("After loading XFeat")

    # Load vocabulary and map
    test_dataset_path = 'resources/tum_fr1/images'
    vocabulary = 'resources/tum_fr1/vocabularies/vocab_tree.bin'
    
    data = np.load('resources/tum_fr1/map_databases/tumfr1_map_train.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    # Camera parameters
    camera_params = load_camera_params(CAMERA_PARAMS_PATH)
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

    # Get test frames
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[500:]  # Use images after first 500
    print(f"Map built with first 500 images")
    print(f"Testing with {len(test_frames)} remaining images")

    # Feature counts to test
    feature_counts = [50, 100, 200, 500, 1000, 1500, 2000]
    
    # Storage for all results
    all_results = {}
    
    # Run experiment for each feature count
    for N in feature_counts:
        print("\n" + "="*60)
        print(f"TESTING WITH N={N} FEATURES")
        print("="*60)
        
        errors = []
        match_counts = []
        timings = {'extract': [], 'match': [], 'pnp': []}
        
        # Counters
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
            
            # Extract features with current N
            t_start = time.time()
            with torch.no_grad():
                output = xfeat.detectAndCompute(frame_gray, top_k=N)
            t_extract = time.time() - t_start
            
            features = output[0]
            keypoints = features['keypoints'].cpu().numpy()
            descriptors = features['descriptors'].cpu().numpy()
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
            
            if len(matched_3d) < 4:
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
            
            if not success:
                pnp_failed += 1
                continue
            
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
            else:
                rejected_low_inliers += 1
        
        # Compute statistics for this N
        if len(errors) > 0:
            median_error = np.median(errors) * 100  # cm
            mean_error = np.mean(errors) * 100  # cm
            success_rate = len(errors) / len(test_frames) * 100
            
            mean_extract = np.mean(timings['extract']) * 1000  # ms
            mean_match = np.mean(timings['match']) * 1000  # ms
            mean_pnp = np.mean(timings['pnp']) * 1000  # ms
            mean_total = mean_extract + mean_match + mean_pnp
            fps = 1000.0 / mean_total
            
            all_results[N] = {
                'errors': np.array(errors),
                'match_counts': np.array(match_counts),
                'timings_extract': np.array(timings['extract']),
                'timings_match': np.array(timings['match']),
                'timings_pnp': np.array(timings['pnp']),
                'success_rate': len(errors) / len(test_frames),
                'median_error_cm': median_error,
                'mean_error_cm': mean_error,
                'fps': fps,
                'mean_extract_ms': mean_extract,
                'mean_match_ms': mean_match,
                'mean_pnp_ms': mean_pnp,
                'mean_total_ms': mean_total,
                'N': N
            }
            
            # Save individual result file
            np.savez(f'results/fr1_fp32_N{N}_errors.npz',
                     errors=np.array(errors),
                     match_counts=np.array(match_counts),
                     timings_extract=np.array(timings['extract']),
                     timings_match=np.array(timings['match']),
                     timings_pnp=np.array(timings['pnp']),
                     success_rate=len(errors)/len(test_frames),
                     N=N)
            
            print(f"\nResults for N={N}:")
            print(f"  Success rate:   {success_rate:.1f}%")
            print(f"  Median error:   {median_error:.2f} cm")
            print(f"  Mean error:     {mean_error:.2f} cm")
            print(f"  FPS:            {fps:.2f}")
            print(f"  Timing breakdown:")
            print(f"    Extract: {mean_extract:.2f} ms")
            print(f"    Match:   {mean_match:.2f} ms")
            print(f"    PnP:     {mean_pnp:.2f} ms")
            print(f"    Total:   {mean_total:.2f} ms")
        else:
            print(f"\n❌ No successful localizations for N={N}")
    
    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'N':>6} | {'Success%':>8} | {'Median(cm)':>10} | {'FPS':>6} | {'Extract':>8} | {'Match':>7} | {'PnP':>6}")
    print("-"*70)
    
    for N in feature_counts:
        if N in all_results:
            r = all_results[N]
            print(f"{N:6d} | {r['success_rate']*100:7.1f}% | {r['median_error_cm']:10.2f} | "
                  f"{r['fps']:6.2f} | {r['mean_extract_ms']:7.2f}ms | "
                  f"{r['mean_match_ms']:6.2f}ms | {r['mean_pnp_ms']:5.2f}ms")
    
    print("\n✓ Results saved to results/fr1_fp32_N*_errors.npz")
    print("="*60)
