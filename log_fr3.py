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
import yaml
from test.memory import MemoryMonitor
import glob

def compute_image_quality_features(frame_gray):
    """
    Compute image quality metrics for a grayscale frame.
    
    Args:
        frame_gray: Grayscale image (H x W) numpy array
        
    Returns:
        dict with image quality features
    """
    features = {}
    
    # 1. Blur score (Laplacian variance)
    # Higher = sharper image
    laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
    features['blur_score'] = laplacian.var()
    
    # 2. Brightness (mean intensity)
    features['brightness'] = np.mean(frame_gray)
    
    # 3. Contrast (std intensity)
    features['contrast'] = np.std(frame_gray)
    
    # 4. Edge density (Canny edges)
    edges = cv2.Canny(frame_gray, 50, 150)
    features['edge_density'] = np.sum(edges > 0) / edges.size
    
    # 5. Histogram uniformity (entropy of intensity distribution)
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]  # Remove zero bins
    entropy = -np.sum(hist * np.log2(hist))
    features['histogram_uniformity'] = entropy / 8.0  # Normalize
    
    return features

def compute_geometric_features(inlier_points_2d, inlier_points_3d, camera_matrix, R, t, image_shape):
    """
    Compute geometric quality features from match configuration.
    
    Args:
        inlier_points_2d: (N, 2) array of 2D keypoints
        inlier_points_3d: (N, 3) array of 3D points
        camera_matrix: 3x3 camera intrinsic matrix
        R: 3x3 rotation matrix
        t: (3,) translation vector
        image_shape: (height, width) tuple
        
    Returns:
        dict with geometric features
    """
    n_inliers = len(inlier_points_2d)
    height, width = image_shape
    diagonal = np.sqrt(height**2 + width**2)
    
    features = {}
    
    # ========== TIER 1 FEATURES ==========
    
    # 1. Number of inliers (already have, but include for completeness)
    features['n_inliers'] = n_inliers
    
    # 2. Match spatial spread (normalized by image diagonal)
    if n_inliers >= 3:
        # Compute minimum enclosing circle
        center, radius = cv2.minEnclosingCircle(inlier_points_2d.astype(np.float32))
        features['match_spread_normalized'] = radius / diagonal
        
        # Alternative: Standard deviation of positions
        features['match_std_x'] = np.std(inlier_points_2d[:, 0]) / width
        features['match_std_y'] = np.std(inlier_points_2d[:, 1]) / height
    else:
        features['match_spread_normalized'] = 0.0
        features['match_std_x'] = 0.0
        features['match_std_y'] = 0.0
    
    # ========== TIER 2 FEATURES ==========
    
    # 3. Depth variance (relative to mean depth)
    # Transform 3D points to camera frame
    points_cam = (R @ inlier_points_3d.T).T + t.reshape(1, 3)
    depths = points_cam[:, 2]  # Z coordinate is depth
    
    if len(depths) > 1 and np.mean(depths) > 0:
        features['depth_mean'] = np.mean(depths)
        features['depth_std'] = np.std(depths)
        features['depth_relative_std'] = np.std(depths) / (np.mean(depths) + 1e-6)
        features['depth_range'] = (np.max(depths) - np.min(depths)) / (np.mean(depths) + 1e-6)
    else:
        features['depth_mean'] = 0.0
        features['depth_std'] = 0.0
        features['depth_relative_std'] = 0.0
        features['depth_range'] = 0.0
    
    # 4. Match distribution across image quadrants
    # Count how many quadrants have at least 10% of matches
    cx_img = width / 2
    cy_img = height / 2
    
    quadrant_counts = np.zeros(4)
    for pt in inlier_points_2d:
        x, y = pt
        if x < cx_img and y < cy_img:
            quadrant_counts[0] += 1  # Top-left
        elif x >= cx_img and y < cy_img:
            quadrant_counts[1] += 1  # Top-right
        elif x < cx_img and y >= cy_img:
            quadrant_counts[2] += 1  # Bottom-left
        else:
            quadrant_counts[3] += 1  # Bottom-right
    
    quadrant_fractions = quadrant_counts / (n_inliers + 1e-6)
    features['n_quadrants_active'] = np.sum(quadrant_fractions >= 0.1)
    features['quadrant_entropy'] = -np.sum(quadrant_fractions * np.log(quadrant_fractions + 1e-6))
    
    # ========== TIER 3 FEATURES ==========
    
    # 5. Viewing angles (simplified: just use depth as proxy)
    # More sophisticated: compute angle between camera ray and point normal
    # For now, use inverse depth as proxy (closer points = better angles)
    if len(depths) > 0:
        features['mean_inverse_depth'] = np.mean(1.0 / (depths + 1e-6))
    else:
        features['mean_inverse_depth'] = 0.0
    
    # 6. Condition number estimate
    # Use spatial spread in X and Y as proxy for condition number
    # Well-conditioned: spread in both X and Y
    # Ill-conditioned: spread only in one direction
    if n_inliers >= 3:
        cov = np.cov(inlier_points_2d.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[1] > 1e-6:
            features['condition_estimate'] = eigenvalues[0] / eigenvalues[1]
        else:
            features['condition_estimate'] = 1e6  # Essentially degenerate
    else:
        features['condition_estimate'] = 1e6
    
    return features


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

    test_dataset_path = 'resources/tum_fr3/images'

    vocabulary = 'resources/tum_fr3/vocabularies/vocab_tree.bin'
    print("Loaded existing vocabulary")

    data = np.load('resources/tum_fr3/map_databases/tum_fr3_train.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']

    # Load quality metrics (with backward compatibility)
    if 'track_lengths' in data and 'ba_errors' in data:
        map_track_lengths = data['track_lengths']
        map_ba_errors = data['ba_errors']
        print(f"Loaded map with quality metrics:")
        print(f"  Points: {len(map_3d_points)}")
        print(f"  Mean track length: {np.mean(map_track_lengths):.1f}")
        print(f"  Mean BA error: {np.mean(map_ba_errors):.4f}")
        has_quality_metrics = True
    else:
        print(f"Loaded map: {len(map_3d_points)} points (no quality metrics)")
        map_track_lengths = None
        map_ba_errors = None
        has_quality_metrics = False


    MemoryMonitor.print_memory("After loading map")

    camera_params = load_camera_params(CAMERA_PARAMS_PATH)
    print(f"Loaded camera: {camera_params['width']}x{camera_params['height']}")
    print(f"  fx={camera_params['fx']:.2f}, fy={camera_params['fy']:.2f}")
    print(f"  cx={camera_params['cx']:.2f}, cy={camera_params['cy']:.2f}")

    K_cv = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    print("\nBuilding vocabulary matcher...")
    t_start = time.time()
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")

    MemoryMonitor.print_memory("After building matcher")

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

    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION TEST - TUM FR3 ")
    print("="*60)
    
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[0:]
    
    
    print(f"Testing with {len(test_frames)} remaining images\n")
    
    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}
    pose_uncertainties = []
    geometric_features_log = []

    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    rejected_reproj_error = 0

    
    for i, frame_path in enumerate(test_frames):
        frame_name = os.path.basename(frame_path)
        
        if frame_name not in gt_poses:
            continue
        
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ========== COMPUTE IMAGE QUALITY FEATURES ==========
        img_quality = compute_image_quality_features(frame_gray)

        
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=200)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        t_start = time.time()

        query_idx_all, map_idx_all, distances_all, ranks_all = matcher.match_with_stats(
            descriptors, k_nearest_words=3
        )

        t_match = time.time() - t_start

        from collections import defaultdict

        query_matches_all = defaultdict(list)
        for q_idx, m_idx, dist, rank in zip(query_idx_all, map_idx_all, distances_all, ranks_all):
            query_matches_all[q_idx].append((m_idx, dist, rank))

        match_stats = {}
        for q_idx, matches in query_matches_all.items():
            best_map_idx, best_distance, _ = matches[0]
            
            if len(matches) > 1:
                second_best_distance = matches[1][1]
                ratio_test_value = best_distance / (second_best_distance + 1e-8)
            else:
                ratio_test_value = 1.0
            
            match_stats[q_idx] = {
                'best_map_idx': best_map_idx,
                'best_distance': best_distance,
                'ratio_test_value': ratio_test_value,
                'n_candidates': len(matches)
            }

        query_to_map = {}
        for q_idx, stats in match_stats.items():
            if stats['ratio_test_value'] < 0.80:
                if q_idx not in query_to_map or stats['best_distance'] < query_to_map[q_idx][1]:
                    query_to_map[q_idx] = (stats['best_map_idx'], stats['best_distance'])

        matched_3d = []
        matched_2d = []
        matched_query_indices = []
        matched_map_indices = []  # NEW: Track which map points were matched
        
        for q_idx, (m_idx, dist) in query_to_map.items():
            matched_3d.append(map_3d_points[m_idx])
            matched_2d.append(keypoints[q_idx])
            matched_query_indices.append(q_idx)
            matched_map_indices.append(m_idx)  
        
        if len(matched_3d) < 4:
            skipped_too_few_matches += 1
            continue
            
        matched_3d = np.array(matched_3d, dtype=np.float32)
        matched_2d = np.array(matched_2d, dtype=np.float32)
        
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
        
        timings['extract'].append(t_extract)
        timings['match'].append(t_match)
        timings['pnp'].append(t_pnp)
        
        if not success:
            pnp_failed += 1
            continue
        
        if success and len(inliers) >= 6:
            inlier_points_3d = matched_3d[inliers.flatten()]
            inlier_points_2d = matched_2d[inliers.flatten()]
            
            projected, _ = cv2.projectPoints(inlier_points_3d, rvec, tvec, K_cv, dist_coeffs)
            reproj_errors = np.linalg.norm(inlier_points_2d - projected.squeeze(), axis=1)
            median_reproj_error = np.median(reproj_errors)
            
            if median_reproj_error > 5.0:
                rejected_reproj_error += 1
                continue


            # NEW SECTION: Compute aggregate quality metrics for inlier map points
            map_quality_features = {}

            if has_quality_metrics:
                # Get map indices for the inliers
                inlier_indices_flat = inliers.flatten()
                inlier_map_idx = [matched_map_indices[i] for i in inlier_indices_flat]


                # Look up quality metrics for these map points
                inlier_track_lengths = map_track_lengths[inlier_map_idx]
                inlier_ba_errors = map_ba_errors[inlier_map_idx]
                
                # Compute aggregate statistics
                map_quality_features['mean_inlier_track_length'] = float(np.mean(inlier_track_lengths))
                map_quality_features['median_inlier_track_length'] = float(np.median(inlier_track_lengths))
                map_quality_features['mean_inlier_ba_error'] = float(np.mean(inlier_ba_errors))
                map_quality_features['median_inlier_ba_error'] = float(np.median(inlier_ba_errors))

                # Fraction of high-quality inliers (track length >= 5)
                map_quality_features['frac_high_quality_inliers'] = float(np.sum(inlier_track_lengths >= 5) / len(inlier_track_lengths))
            else:
                # Default values if no quality metrics available
                map_quality_features['mean_inlier_track_length'] = 0.0
                map_quality_features['median_inlier_track_length'] = 0.0
                map_quality_features['mean_inlier_ba_error'] = 0.0
                map_quality_features['median_inlier_ba_error'] = 0.0
                map_quality_features['frac_high_quality_inliers'] = 0.0


            
            R_cam, _ = cv2.Rodrigues(rvec)
            C_colmap = -R_cam.T @ tvec.flatten()
            C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
            
            # ========== COMPUTE GEOMETRIC FEATURES ==========
            geom_features = compute_geometric_features(
                inlier_points_2d=inlier_points_2d,
                inlier_points_3d=inlier_points_3d,
                camera_matrix=K_cv,
                R=R_cam,
                t=tvec.flatten(),
                image_shape=(frame_gray.shape[0], frame_gray.shape[1])
            )

            # Add image quality features
            for key, val in img_quality.items():
                geom_features[f'img_{key}'] = val
            
            # Log geometric features + image quality
            geom_features['frame'] = frame_name
            geometric_features_log.append(geom_features)
            
            # ========== UNCERTAINTY ESTIMATION WITH GEOMETRIC FEATURES ==========
            if uncertainty_estimator is not None:
                inlier_indices_flat = inliers.flatten()
                inlier_best_distances = []
                inlier_ratio_values = []
                inlier_n_candidates = []
                
                for inlier_idx in inlier_indices_flat:
                    q_idx = matched_query_indices[inlier_idx]
                    stats = match_stats[q_idx]
                    
                    inlier_best_distances.append(stats['best_distance'])
                    inlier_ratio_values.append(stats['ratio_test_value'])
                    inlier_n_candidates.append(stats['n_candidates'])
                
                inlier_best_distances = np.array(inlier_best_distances)
                inlier_ratio_values = np.array(inlier_ratio_values)
                inlier_n_candidates = np.array(inlier_n_candidates)
                
                detector_scores = np.full(len(inlier_indices_flat), 0.25)
                
                # Initialize to None in case of failure
                uncertainty_result = None
                
                try:

                    # Merge geometric and map quality features
                    combined_features = {**geom_features, **map_quality_features}

                    uncertainty_result = uncertainty_estimator.estimate_pose_uncertainty(
                        points_3d=inlier_points_3d,
                        points_2d=inlier_points_2d,
                        camera_matrix=K_cv,
                        R=R_cam,
                        t=tvec.flatten(),
                        best_match_distances=inlier_best_distances,
                        ratio_test_values=inlier_ratio_values,
                        n_candidates=inlier_n_candidates,
                        reprojection_errors=reproj_errors,
                        detector_scores=detector_scores,
                        geometric_features=combined_features,  # Pass geometric features
                        filter_low_confidence=False
                    )
                    
                    # Store uncertainty + geometric features
                    unc_entry = {
                        'frame': frame_name,
                        'rotation_std': uncertainty_result['info']['rotation_uncertainty'],
                        'translation_std': uncertainty_result['info']['translation_uncertainty'],
                        'mean_confidence': np.mean(uncertainty_result['confidence']),
                        'mean_pixel_uncertainty': uncertainty_result['info']['mean_pixel_uncertainty'],
                        'mean_ratio_test': np.mean(inlier_ratio_values),
                        'mean_n_candidates': np.mean(inlier_n_candidates)
                    }
                    
                    # Add geometric features to uncertainty entry
                    for key, val in combined_features.items():
                        if key != 'frame':
                            unc_entry[f'geom_{key}'] = val
                    
                    pose_uncertainties.append(unc_entry)
                    
                except Exception as e:
                    print(f"Warning: Uncertainty estimation failed for frame {i+1}: {e}")
            
            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            errors.append(error_meters)
            match_counts.append(len(inliers))
            
            if (i + 1) % 20 == 0:
                uncertainty_str = ""
                if uncertainty_result is not None:
                    conf = np.mean(uncertainty_result['confidence'])
                    sigma = uncertainty_result['info']['mean_pixel_uncertainty']
                    spread = geom_features['match_spread_normalized']
                    uncertainty_str = f", Conf={conf:.3f}, σ={sigma:.2f}px, Spread={spread:.2f}"
                
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}"
                      f"{uncertainty_str}")
        else:
            rejected_low_inliers += 1

    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION RESULTS (WITH GEOMETRIC FEATURES)")
    print("="*60)
    print(f"Total frames attempted: {len(test_frames)}")
    print(f"Successful localizations: {len(errors)}")
    print(f"Success rate: {len(errors)/len(test_frames)*100:.1f}%")
    
    if len(errors) > 0:
        print(f"\nLocalization Accuracy:")
        print(f"  Mean error:   {np.mean(errors):.4f} m ({np.mean(errors)*100:.2f} cm)")
        print(f"  Median error: {np.median(errors):.4f} m ({np.median(errors)*100:.2f} cm)")
        print(f"  Std error:    {np.std(errors):.4f} m")
        
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
            
            # Print geometric feature statistics
            if len(geometric_features_log) > 0:
                print(f"\nGeometric Feature Statistics:")
                print(f"  Match spread (normalized):  {np.mean([g['match_spread_normalized'] for g in geometric_features_log]):.3f} ± {np.std([g['match_spread_normalized'] for g in geometric_features_log]):.3f}")
                print(f"  Depth relative std:         {np.mean([g['depth_relative_std'] for g in geometric_features_log]):.3f} ± {np.std([g['depth_relative_std'] for g in geometric_features_log]):.3f}")
                print(f"  Active quadrants:           {np.mean([g['n_quadrants_active'] for g in geometric_features_log]):.2f} ± {np.std([g['n_quadrants_active'] for g in geometric_features_log]):.2f}")

        # Save results with geometric features
        save_dict = {
            'errors': np.array(errors),
            'match_counts': np.array(match_counts),
            'timings_extract': np.array(timings['extract']),
            'timings_match': np.array(timings['match']),
            'timings_pnp': np.array(timings['pnp']),
            'success_rate': len(errors)/len(test_frames)
        }
        
        if len(pose_uncertainties) > 0:
            save_dict['uncertainty_data'] = pose_uncertainties
        
        
        
        # Save detailed CSV with geometric features
        if len(pose_uncertainties) > 0:
            import pandas as pd
            
            # Build comprehensive DataFrame
            df_data = {
                'frame': [u['frame'] for u in pose_uncertainties],
                'error_m': errors[:len(pose_uncertainties)],
                'n_matches': match_counts[:len(pose_uncertainties)],
                'mean_confidence': [u['mean_confidence'] for u in pose_uncertainties],
                'mean_pixel_uncertainty_px': [u['mean_pixel_uncertainty'] for u in pose_uncertainties],
                'mean_ratio_test': [u.get('mean_ratio_test', 0.5) for u in pose_uncertainties],
                'mean_n_candidates': [u.get('mean_n_candidates', 1.0) for u in pose_uncertainties],
                'translation_std_total_m': np.linalg.norm([u['translation_std'] for u in pose_uncertainties], axis=1),
            }
            
            # Add all geometric features
            geom_feature_names = [
                'n_inliers', 'match_spread_normalized', 'match_std_x', 'match_std_y',
                'depth_mean', 'depth_std', 'depth_relative_std', 'depth_range',
                'n_quadrants_active', 'quadrant_entropy', 'mean_inverse_depth', 'condition_estimate',
                'mean_inlier_track_length', 'median_inlier_track_length',
                'mean_inlier_ba_error', 'median_inlier_ba_error',
                'frac_high_quality_inliers', 
                # Image quality features
                'img_blur_score', 'img_brightness', 'img_contrast', 
                'img_edge_density', 'img_histogram_uniformity'

            ]
            
            for feat_name in geom_feature_names:
                key = f'geom_{feat_name}'
                if key in pose_uncertainties[0]:
                    df_data[feat_name] = [u[key] for u in pose_uncertainties]
            
            uncertainty_df = pd.DataFrame(df_data)
            
            uncertainty_df.to_csv('results/fr3_uncertainty_with_geometry.csv', index=False)
            print("✓ Saved detailed data to results/xxx.csv")
            print(f"  Total columns: {len(uncertainty_df.columns)}")
            print(f"  Geometric features included: {len(geom_feature_names)}")
            
            # Quick correlation check
            if 'match_spread_normalized' in uncertainty_df.columns:
                from scipy.stats import pearsonr
                
                r_spread, p_spread = pearsonr(uncertainty_df['match_spread_normalized'], uncertainty_df['error_m'])
                r_depth, p_depth = pearsonr(uncertainty_df['depth_relative_std'], uncertainty_df['error_m'])
                r_quad, p_quad = pearsonr(uncertainty_df['n_quadrants_active'], uncertainty_df['error_m'])
                
                print(f"\n  Quick correlation with error:")
                print(f"    Match spread:       r={r_spread:+.3f}, p={p_spread:.2e}")
                print(f"    Depth variance:     r={r_depth:+.3f}, p={p_depth:.2e}")
                print(f"    Active quadrants:   r={r_quad:+.3f}, p={p_quad:.2e}")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)