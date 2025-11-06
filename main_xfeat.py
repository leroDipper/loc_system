import torch
import os
from accelerated_modules import vocab_match
from loc_modules.load_gt_params import GroundTruthParams
import cv2
import numpy as np
import time
import json
from scipy.spatial.transform import Rotation
from loc_modules import MapLoader, Localiser

if __name__ == "__main__":
    print("Loading COLMAP → Ground Truth transformation...")
    scale, R, t = GroundTruthParams.load_transformation('colmap_database/large_map_xfeat/colmap_to_gt_transform.json')

    #load ground truth positions
    gt_path = 'colmap_database/large_map_xfeat/ground_truth_poses_circ_f8.json'
    gt_poses = GroundTruthParams.load_ground_truth(gt_path)

    xfeat = None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
        xfeat = xfeat.to(device)
        xfeat.eval()
    except Exception as e:
        print(f"Error loading XFeat model: {e}")
        exit()

    test_dataset_path = 'colmap_database/large_map/large_set_test_640x480'

    #load existing vocabulary
    vocabulary = np.load('resources/vocabularies/vocabulary_circ_f8.npy')
    print("Loaded existing vocabulary")

    #load colmap
    data = np.load('resources/map_databases/colmap_map_train_set.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    # Camera parameters from your image
    camera_params = {
        'width': 640,
        'height': 480,
        'fx': 768.0,
        'fy': 768.0,
        'cx': 320.0,
        'cy': 240.0
    }

    K_cv = np.array([[camera_params['fx'], 0, camera_params['cx']],
                    [0, camera_params['fy'], camera_params['cy']],
                    [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # Build matcher
    print("\nBuilding vocabulary matcher...")
    t_start = time.time()
    matcher = vocab_match.VocabMatcher(vocabulary, map_descriptors, top_k=3)
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")

    frame_name = 'frame_0143.jpg'
    frame_path = os.path.join(test_dataset_path, frame_name)

    frame = cv2.imread(frame_path)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    t_start = time.time()
    with torch.no_grad():
        output = xfeat.detectAndCompute(frame_gray, top_k=100)


    t_extract = time.time() - t_start
    features = output[0]
    keypoints = features['keypoints'].cpu().numpy()
    descriptors = features['descriptors'].cpu().numpy()
    print(f"Extracted {len(keypoints)} features in {t_extract:.3f}s")
    descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)

    # Match
    t_start = time.time()
    query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80)
    t_match = time.time() - t_start
    print(f"Matched {len(query_idx)} in {t_match:.3f}s")

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

    if success:
            R_cam, _ = cv2.Rodrigues(rvec)
            C_colmap = -R_cam.T @ tvec.flatten()
            
            # Transform to meters
            C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
            
            # Get ground truth
            C_gt = gt_poses[frame_name]
            
            # Compute errors
            error_colmap_units = np.linalg.norm(C_colmap - C_gt)  # This is wrong! Different coordinate frames
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            print(f"\n{'='*60}")
            print("RESULTS")
            print(f"{'='*60}")
            #print(f"\nEstimated (COLMAP):  {C_colmap}")
            print(f"Estimated (meters) xFeat:  {C_meters}")
            print(f"Ground truth:        {C_gt}")
            print(f"\nError: {error_meters:.4f} meters ({error_meters*100:.2f} cm)")
            print(f"Inliers: {len(inliers)}/{len(matched_3d)}")

            print(f"\n{'='*60}")
            print("TIMING")
            print(f"{'='*60}")
            print(f"Feature extraction: {t_extract:.3f}s")
            print(f"Matching:           {t_match:.3f}s")
            print(f"PnP:                {t_pnp:.3f}s")
            print(f"Total per frame:    {t_extract + t_match + t_pnp:.3f}s")
            print(f"FPS:                {1.0/(t_extract + t_match + t_pnp):.2f}")
    else:
        print("PnP failed to find a solution.")







    











