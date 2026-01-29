import torch
import os
from accelerated_modules import vocab_tree_match
from loc_modules.load_gt_params import GroundTruthParams
from loc_modules.pose_uncertainty import PoseUncertaintyEstimator
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

    # Initialize uncertainty estimator
    print("\nLoading uncertainty estimation model...")
    try:
        uncertainty_estimator = PoseUncertaintyEstimator(
            model_path="results/match_confidence_model.joblib",
            sigma_base=1.0,
            min_confidence=0.3,
            use_reprojection_refinement=True
        )
    except FileNotFoundError:
        print("Warning: Uncertainty model not found. Skipping uncertainty estimation.")
        uncertainty_estimator = None

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
    pose_uncertainties = []  # Store uncertainty estimates

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
        
        # Extract features
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=100)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        # Match
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80, k_nearest_words = 3)
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
            
            # Estimate pose uncertainty
            uncertainty_result = None
            if uncertainty_estimator is not None:
                # Gather match features for inliers
                inlier_indices = inliers.flatten()
                
                # Get descriptor distances for inliers
                inlier_distances = np.array([query_to_map[q_idx][1] for q_idx in 
                                            [list(query_to_map.keys())[i] for i in inlier_indices]])
                
                # Compute ratio test values (use 0.5 as placeholder since we don't store second-best)
                ratio_values = np.full(len(inlier_indices), 0.5)
                
                # Number of candidates (1 per match since we filtered duplicates)
                n_candidates_arr = np.ones(len(inlier_indices))
                
                # Detector scores (get from original features if available)
                detector_scores = np.full(len(inlier_indices), 0.25)
                
                try:
                    uncertainty_result = uncertainty_estimator.estimate_pose_uncertainty(
                        points_3d=inlier_points_3d,
                        points_2d=inlier_points_2d,
                        camera_matrix=K_cv,
                        R=R_cam,
                        t=tvec.flatten(),
                        best_match_distances=inlier_distances,
                        ratio_test_values=ratio_values,
                        n_candidates=n_candidates_arr,
                        reprojection_errors=reproj_errors,
                        detector_scores=detector_scores,
                        filter_low_confidence=False
                    )
                    pose_uncertainties.append({
                        'frame': frame_name,
                        'rotation_std': uncertainty_result['info']['rotation_uncertainty'],
                        'translation_std': uncertainty_result['info']['translation_uncertainty'],
                        'mean_confidence': np.mean(uncertainty_result['confidence']),
                        'mean_pixel_uncertainty': uncertainty_result['info']['mean_pixel_uncertainty']
                    })
                except Exception as e:
                    print(f"Warning: Uncertainty estimation failed for frame {i+1}: {e}")
            
            # Accept localization
            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            errors.append(error_meters)
            match_counts.append(len(inliers))
            previous_C = C_meters
            
            if (i + 1) % 20 == 0:
                uncertainty_str = ""
                if uncertainty_result is not None:
                    conf = np.mean(uncertainty_result['confidence'])
                    sigma = uncertainty_result['info']['mean_pixel_uncertainty']
                    uncertainty_str = f", Conf={conf:.3f}, σ={sigma:.2f}px"
                
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}"
                      f"{uncertainty_str}")
        else:
            rejected_low_inliers += 1

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
        
        # Print uncertainty statistics if available
        if len(pose_uncertainties) > 0:
            mean_confs = [u['mean_confidence'] for u in pose_uncertainties]
            mean_sigmas = [u['mean_pixel_uncertainty'] for u in pose_uncertainties]
            rot_stds = np.array([u['rotation_std'] for u in pose_uncertainties])
            trans_stds = np.array([u['translation_std'] for u in pose_uncertainties])
            
            print(f"\nPose Uncertainty Estimates:")
            print(f"  Match confidence:       {np.mean(mean_confs):.3f} ± {np.std(mean_confs):.3f}")
            print(f"  Pixel uncertainty:      {np.mean(mean_sigmas):.2f} ± {np.std(mean_sigmas):.2f} px")
            print(f"  Rotation std (mean):    [{np.mean(rot_stds[:,0]):.4f}, {np.mean(rot_stds[:,1]):.4f}, {np.mean(rot_stds[:,2]):.4f}] rad")
            print(f"  Translation std (mean): [{np.mean(trans_stds[:,0]):.4f}, {np.mean(trans_stds[:,1]):.4f}, {np.mean(trans_stds[:,2]):.4f}] m")

        # Save results
        save_dict = {
            'errors': np.array(errors),
            'match_counts': np.array(match_counts),
            'timings_extract': np.array(timings['extract']),
            'timings_match': np.array(timings['match']),
            'timings_pnp': np.array(timings['pnp']),
            'success_rate': len(errors)/len(test_frames)
        }
        
        # # Add uncertainty data if available
        # if len(pose_uncertainties) > 0:
        #     save_dict['uncertainty_data'] = pose_uncertainties
        
        # np.savez('results/fr1_with_uncertainty.npz', **save_dict)
        # print("✓ Saved results to results/fr1_with_uncertainty.npz")
        
        # Save uncertainty data as CSV for easy viewing
        if len(pose_uncertainties) > 0:
            import pandas as pd
            
            # Flatten uncertainty data into a DataFrame
            uncertainty_df = pd.DataFrame({
                'frame': [u['frame'] for u in pose_uncertainties],
                'error_m': errors[:len(pose_uncertainties)],
                'n_matches': match_counts[:len(pose_uncertainties)],
                'mean_confidence': [u['mean_confidence'] for u in pose_uncertainties],
                'mean_pixel_uncertainty_px': [u['mean_pixel_uncertainty'] for u in pose_uncertainties],
                'rotation_std_x_rad': [u['rotation_std'][0] for u in pose_uncertainties],
                'rotation_std_y_rad': [u['rotation_std'][1] for u in pose_uncertainties],
                'rotation_std_z_rad': [u['rotation_std'][2] for u in pose_uncertainties],
                'translation_std_x_m': [u['translation_std'][0] for u in pose_uncertainties],
                'translation_std_y_m': [u['translation_std'][1] for u in pose_uncertainties],
                'translation_std_z_m': [u['translation_std'][2] for u in pose_uncertainties],
            })
            
            # Add total uncertainties
            uncertainty_df['rotation_std_total_rad'] = np.linalg.norm(uncertainty_df[['rotation_std_x_rad', 'rotation_std_y_rad', 'rotation_std_z_rad']], axis=1)
            uncertainty_df['translation_std_total_m'] = np.linalg.norm(uncertainty_df[['translation_std_x_m', 'translation_std_y_m', 'translation_std_z_m']], axis=1)
            
            uncertainty_df.to_csv('results/uncertainty_estimates.csv', index=False)
            print("✓ Saved uncertainty data to results/uncertainty_estimates.csv")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)