import torch
import os
from accelerated_modules import vocab_tree_match, image_retrieval_match
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
    print(f"Loaded map: {len(map_3d_points)} points across {len(image_names)} images")

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
    print("INT8 QUERIES vs FP32 MAP - EUROC mh_01")
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
    print(f"COLMAP reconstructed {len(colmap_set)} total images")  # NEW LINE
    print(f"Testing with {len(test_images)} held-out images (INT8)\n")
    
    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}

    skipped_no_gt = 0
    skipped_no_image = 0
    skipped_too_few_matches = 0
    pnp_failed = 0
    rejected_low_inliers = 0
    rejected_reproj_error = 0

    test_image_list = sorted(list(test_images))
    
    for i, frame_path in enumerate(test_image_list):
        frame_name = os.path.basename(frame_path)
        
        if frame_name not in gt_poses:
            skipped_no_gt += 1
            continue
        
        frame = cv2.imread(frame_path)
        if frame is None:
            skipped_no_image += 1
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=250)
        t_extract = time.time() - t_start
        
        features = output[0]
        keypoints = features['keypoints'].cpu().numpy()
        descriptors = features['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)
        
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80)
        t_match = time.time() - t_start
        
        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)
        
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
        
        if not success or len(inliers) < 6:
            if not success:
                pnp_failed += 1
            else:
                rejected_low_inliers += 1
            continue
            
        inlier_points_3d = matched_3d[inliers.flatten()]
        inlier_points_2d = matched_2d[inliers.flatten()]
        
        projected, _ = cv2.projectPoints(inlier_points_3d, rvec, tvec, K_cv, dist_coeffs)
        reproj_errors = np.linalg.norm(inlier_points_2d - projected.squeeze(), axis=1)
        median_reproj_error = np.median(reproj_errors)
        
        if median_reproj_error > 5.0:
            rejected_reproj_error += 1
            continue
        
        R_cam, _ = cv2.Rodrigues(rvec)
        C_colmap = -R_cam.T @ tvec.flatten()
        C_metres = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
        
        C_gt = gt_poses[frame_name]
        error_metres = np.linalg.norm(C_metres - C_gt)
        
        errors.append(error_metres)
        match_counts.append(len(inliers))
        
        if (i + 1) % 50 == 0:
            print(f"Frame {i+1}/{len(test_image_list)}: "
                  f"Error={error_metres:.3f}m, "
                  f"Matches={len(inliers)}/{len(matched_3d)}")
            MemoryMonitor.print_memory(f"After {i+1} frames")

    
    print("\n" + "="*60)
    print("FAILURE BREAKDOWN")
    print("="*60)
    print(f"Skipped - no ground truth:      {skipped_no_gt}")
    print(f"Skipped - image failed to load: {skipped_no_image}")
    print(f"Skipped - too few matches (<4): {skipped_too_few_matches}")
    print(f"PnP RANSAC failed:              {pnp_failed}")
    print(f"Rejected - low inliers (<6):    {rejected_low_inliers}")
    print(f"Rejected - high reproj. error:  {rejected_reproj_error}")
    print(f"Total failures:                 {len(test_image_list) - len(errors)}")
    print(f"Sanity check (should = {len(test_image_list)}): {len(errors) + skipped_no_gt + skipped_no_image + skipped_too_few_matches + pnp_failed + rejected_low_inliers + rejected_reproj_error}")
    
    print("\n" + "="*60)
    print("LOCALIZATION ACCURACY")
    print("="*60)
    print(f"Total test frames: {(len(test_image_list)-skipped_no_gt)}")
    print(f"Successful localizations: {len(errors)}")
    print(f"Success rate: {len(errors)/(len(test_image_list)-skipped_no_gt)*100:.1f}%")
    print("="*60)
    
    if len(errors) > 0:
        print(f"\nLocalization Accuracy:")
        print(f"  Mean error:   {np.mean(errors):.4f} m ({np.mean(errors)*100:.2f} cm)")
        print(f"  Median error: {np.median(errors):.4f} m ({np.median(errors)*100:.2f} cm)")
        print(f"  Std error:    {np.std(errors):.4f} m")
        print(f"  Max error:    {np.max(errors):.4f} m")
        print(f"  Min error:    {np.min(errors):.4f} m")
        
        below_20cm = np.sum(np.array(errors) <= 0.20)
        print(f"  Frames within 20cm: {below_20cm}/{len(errors)} ({below_20cm/len(errors)*100:.1f}%)")
        
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

        # np.savez('results/mh_01_fp32_errors.npz', 
        #     errors=np.array(errors),
        #     match_counts=np.array(match_counts),
        #     timings_extract=np.array(timings['extract']),
        #     timings_match=np.array(timings['match']),
        #     timings_pnp=np.array(timings['pnp']),
        #     success_rate=len(errors)/len(test_image_list))
        # print("✓ Saved errors to results/mh_01_fp32_errors.npz")
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)