import torch
import os
from accelerated_modules import vocab_tree_match, image_retrieval_match
from loc_modules.load_gt_params import GroundTruthParams
import cv2
import numpy as np
import time
from scipy.spatial.transform import Rotation
import yaml
from test.memory import MemoryMonitor
import glob

def compute_image_quality_features(frame_gray):
    """
    Compute image quality metrics for a grayscale frame.
    """
    features = {}
    
    # 1. Blur score (Laplacian variance)
    laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
    features['img_blur_score'] = laplacian.var()
    
    # 2. Brightness (mean intensity)
    features['img_brightness'] = np.mean(frame_gray)
    
    # 3. Contrast (std intensity)
    features['img_contrast'] = np.std(frame_gray)
    
    # 4. Edge density (Canny edges)
    edges = cv2.Canny(frame_gray, 50, 150)
    features['img_edge_density'] = np.sum(edges > 0) / edges.size
    
    # 5. Histogram uniformity (entropy of intensity distribution)
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    features['img_histogram_uniformity'] = entropy / 8.0
    
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
    
    # 1. Number of inliers
    features['n_inliers'] = n_inliers
    
    # 2. Match spatial spread (normalized by image diagonal)
    if n_inliers >= 3:
        center, radius = cv2.minEnclosingCircle(inlier_points_2d.astype(np.float32))
        features['match_spread_normalized'] = radius / diagonal
        
        features['match_std_x'] = np.std(inlier_points_2d[:, 0]) / width
        features['match_std_y'] = np.std(inlier_points_2d[:, 1]) / height
    else:
        features['match_spread_normalized'] = 0.0
        features['match_std_x'] = 0.0
        features['match_std_y'] = 0.0
    
    # ========== TIER 2 FEATURES ==========
    
    # 3. Depth variance (relative to mean depth)
    points_cam = (R @ inlier_points_3d.T).T + t.reshape(1, 3)
    depths = points_cam[:, 2]
    
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
    
    # 5. Mean inverse depth
    if len(depths) > 0:
        features['mean_inverse_depth'] = np.mean(1.0 / (depths + 1e-6))
    else:
        features['mean_inverse_depth'] = 0.0
    
    # 6. Condition number estimate
    if n_inliers >= 3:
        cov = np.cov(inlier_points_2d.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        if eigenvalues[1] > 1e-6:
            features['condition_estimate'] = eigenvalues[0] / eigenvalues[1]
        else:
            features['condition_estimate'] = 1e6
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
    scale, R, t = GroundTruthParams.load_transformation('resources/mh_01/project_files/colmap_to_gt_transform.json')
    CAMERA_PARAMS_PATH = 'resources/mh_01/images/camera_rectified.yaml'
    EUROC_DATASET_PATH = 'resources/mh_01'
    N_TRAIN_IMAGES = 2900
    test_dataset_path = 'resources/mh_01/images'

    gt_poses = GroundTruthParams.load_euroc_ground_truth_by_image(
        gt_csv_path=os.path.join(EUROC_DATASET_PATH, 'data.csv'),
        image_dir=os.path.join(EUROC_DATASET_PATH, 'images')
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

    vocabulary = 'resources/mh_01/vocabularies/vocab_tree_master.bin'
    print("Loaded existing vocabulary")

    data = np.load('resources/mh_01/map_databases/mh_01_master.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    map_image_ids = data['image_ids']
    image_names = data['image_names']
   
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

    print("\nBuilding image retrieval matcher...")
    t_start = time.time()
    matcher = image_retrieval_match.ImageRetrievalMatcher(
        vocabulary, map_descriptors, map_image_ids, len(image_names)
    )
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")
    MemoryMonitor.print_memory("After building matcher")

    print("\n" + "="*60)
    print("CONTINUOUS LOCALISATION TEST - mh_01 SEQUENCE")
    print("="*60)

    # Load COLMAP reconstructed images
    colmap_images = load_colmap_image_names('resources/mh_01/project_files/images.txt')
    colmap_set = set(colmap_images)

    # Get chronological order
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    all_filenames = [os.path.basename(f) for f in all_frames]

    # Filter to only reconstructed images
    reconstructed_chrono = [f for f in all_filenames if f in colmap_set]

    # Take remaining after first N_TRAIN_IMAGES
    test_images = [os.path.join(test_dataset_path, f) for f in reconstructed_chrono[N_TRAIN_IMAGES:]]

    print(f"Map built with {N_TRAIN_IMAGES} images (FP32)")
    print(f"COLMAP reconstructed {len(colmap_set)} total images")
    print(f"Testing with {len(test_images)} held-out images (INT8)\n")

    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}
    results_log = []  # Stores per-frame features + error for CSV

    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    rejected_reproj_error = 0

    for i, frame_path in enumerate(test_images):
        frame_name = os.path.basename(frame_path)

        if frame_name not in gt_poses:
            skipped_no_gt += 1
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            skipped_no_image += 1
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ========== COMPUTE IMAGE QUALITY FEATURES ==========
        img_quality_features = compute_image_quality_features(frame_gray)

        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=250)
        t_extract = time.time() - t_start

        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)

        t_start = time.time()
        query_idx_all, map_idx_all, distances_all, ranks_all = matcher.match_with_stats(
            descriptors, top_k_images = 10
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
        matched_map_indices = []

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

            # ========== MAP QUALITY FEATURES ==========
            map_quality_features = {}
            if has_quality_metrics:
                inlier_indices_flat = inliers.flatten()
                inlier_map_idx = [matched_map_indices[i] for i in inlier_indices_flat]
                inlier_track_lengths = map_track_lengths[inlier_map_idx]
                inlier_ba_errors = map_ba_errors[inlier_map_idx]

                map_quality_features['mean_inlier_track_length'] = float(np.mean(inlier_track_lengths))
                map_quality_features['median_inlier_track_length'] = float(np.median(inlier_track_lengths))
                map_quality_features['mean_inlier_ba_error'] = float(np.mean(inlier_ba_errors))
                map_quality_features['median_inlier_ba_error'] = float(np.median(inlier_ba_errors))
                map_quality_features['frac_high_quality_inliers'] = float(np.sum(inlier_track_lengths >= 5) / len(inlier_track_lengths))
            else:
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

            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)

            errors.append(error_meters)
            match_counts.append(len(inliers))


            # ========== COMPUTE PHYSICS-BASED UNCERTAINTY ==========
            try:
                n_pts = len(inlier_points_3d)
                pts_cam = (R_cam @ inlier_points_3d.T).T + tvec.flatten()
                pts_centered = pts_cam - pts_cam.mean(axis=0)
                _, s, _ = np.linalg.svd(pts_centered, full_matrices=False)
                condition_3d = float(s[0] / (s[2] + 1e-9))
                depth_condition = float(s[0] / (s[1] + 1e-9))
                fx, fy = K_cv[0,0], K_cv[1,1]
                
                # Build pose Jacobian (2N x 6)
                J = np.zeros((2 * n_pts, 6))
                for i in range(n_pts):
                    X, Y, Z = pts_cam[i]
                    Z2 = Z * Z
                    du_dX, du_dZ = fx / Z, -fx * X / Z2
                    dv_dY, dv_dZ = fy / Z, -fy * Y / Z2
                    J[2*i,   0] = du_dZ*(-Y);  J[2*i,   1] = du_dX*(-Z) + du_dZ*X;  J[2*i,   2] = du_dX*Y
                    J[2*i+1, 0] = dv_dZ*(-Y) + dv_dY*Z;  J[2*i+1, 1] = dv_dZ*X;  J[2*i+1, 2] = dv_dY*(-X)
                    J[2*i,   3] = du_dX;  J[2*i,   4] = 0;       J[2*i,   5] = du_dZ
                    J[2*i+1, 3] = 0;      J[2*i+1, 4] = dv_dY;   J[2*i+1, 5] = dv_dZ

                sigma2 = float(np.mean(reproj_errors**2)) + 1e-6
                H = J.T @ J
                cov_geo = np.linalg.inv(H) * sigma2

                # Per-point map noise
                sigma_point_sq = (inlier_ba_errors ** 2) / np.maximum(inlier_track_lengths, 1)

                M_cov = np.zeros((6, 6))
                for i in range(n_pts):
                    J_i = J[2*i:2*i+2, :]
                    X, Y, Z = pts_cam[i]
                    Z2 = Z * Z
                    J_map_i = np.array([
                        [fx/Z,  0,    -fx*X/Z2],
                        [0,     fy/Z, -fy*Y/Z2]
                    ])
                    reproj_cov_i = sigma_point_sq[i] * (J_map_i @ J_map_i.T)
                    M_cov += J_i.T @ reproj_cov_i @ J_i
                M_cov /= n_pts
                H_inv = cov_geo / sigma2
                cov_map = H_inv @ M_cov @ H_inv
                cov_total = cov_geo + cov_map
                trans_std_total = float(np.sqrt(np.trace(cov_total[3:, 3:])))
                trans_std_crb = float(np.sqrt(np.trace(cov_geo[3:, 3:])) * scale)

                # Translation covariance anisotropy
                trans_cov = cov_total[3:, 3:]
                eigs = np.linalg.eigvalsh(trans_cov)
                eigs = np.abs(eigs) + 1e-12
                trans_anisotropy = float(eigs[2] / eigs[0])
                trans_max_std = float(np.sqrt(eigs[2]))

            except Exception:
                trans_std_total = float(np.mean(reproj_errors))
                trans_std_crb = float(np.mean(reproj_errors))
                
            # ========== BUILD RESULTS LOG ENTRY ==========
            entry = {
                'frame': frame_name,
                'error_m': error_meters,
                'n_matches': len(inliers),
                'mean_inlier_reproj_error': float(np.mean(reproj_errors)),
                'median_inlier_reproj_error': float(np.median(reproj_errors)),
                'std_inlier_reproj_error': float(np.std(reproj_errors)),
                'reproj_error_dist': ','.join(f'{x:.4f}' for x in np.sort(reproj_errors)[:50]),
                'translation_std_total_m': trans_std_total,
                'condition_3d': condition_3d,
                'depth_condition': depth_condition,
                'trans_anisotropy': trans_anisotropy,
                'trans_max_std': trans_max_std,
                'translation_std_crb': trans_std_crb,
                'C_x': float(C_meters[0]),
                'C_y': float(C_meters[1]),
                'C_z': float(C_meters[2])
            }

            



            for key, val in geom_features.items():
                entry[key] = val
            for key, val in map_quality_features.items():
                entry[key] = val
            for key, val in img_quality_features.items():
                entry[key] = val

            results_log.append(entry)

            

            if (i + 1) % 50 == 0:
                print(f"Frame {i+1}/{len(test_images)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}")
        else:
            rejected_low_inliers += 1

    print("\n" + "="*60)
    print("CONTINUOUS LOCALISATION RESULTS")
    print("="*60)
    print(f"Total frames attempted: {(len(test_images) - skipped_no_gt)}")
    print(f"Successful localisations: {len(errors)}")
    print(f"Success rate: {len(errors)/(len(test_images)-skipped_no_gt)*100:.1f}%")

    if len(errors) > 0:
        print(f"\nLocalisation Accuracy:")
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

        # Save detailed CSV with all features
        if len(results_log) > 0:
            import pandas as pd
            os.makedirs('results', exist_ok=True)
            uncertainty_df = pd.DataFrame(results_log)
            uncertainty_df.to_csv('results/mh_01_uncertainty250.csv', index=False)
            print(f"\n✓ Saved detailed data to results/mh_01_uncertainty250.csv")
            print(f"  Rows: {len(uncertainty_df)}, Columns: {len(uncertainty_df.columns)}")

    MemoryMonitor.print_memory("After continuous localisation")

    print("="*60)