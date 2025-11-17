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
from test.memory import MemoryMonitor
import yaml

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

    MemoryMonitor.print_memory("After loading XFeat")

    test_dataset_path = 'colmap_database/large_map/large_set_test_640x480'

    #load existing vocabulary
    vocabulary = 'resources/blender/vocabularies/vocab_tree.bin'
    print("Loaded existing vocabulary")

    #load colmap
    data = np.load('resources/blender/map_databases/colmap_map_train_set.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

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
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)
    t_index = time.time() - t_start
    print(f"Index built in {t_index:.3f}s")

    MemoryMonitor.print_memory("After building matcher")

    # ===================================================================
    # CONTINUOUS LOCALIZATION TEST
    # ===================================================================
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION TEST")
    print("="*60)
    
    # Get all test frames
    import glob
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.jpg")))
    
    # Use first 50 frames (or all if less than 50)
    test_frames = all_frames[:min(50, len(all_frames))]
    print(f"Testing with {len(test_frames)} frames\n")
    
    errors = []
    match_counts = []
    timings = {'extract': [], 'match': [], 'pnp': []}
    
    for i, frame_path in enumerate(test_frames):
        frame_name = os.path.basename(frame_path)
        
        # Read and process frame
        frame = cv2.imread(frame_path)
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
        
        if success and frame_name in gt_poses:
            R_cam, _ = cv2.Rodrigues(rvec)
            C_colmap = -R_cam.T @ tvec.flatten()
            C_meters = GroundTruthParams.colmap_to_meters(C_colmap, scale, R, t)
            C_gt = gt_poses[frame_name]
            error_meters = np.linalg.norm(C_meters - C_gt)
            
            errors.append(error_meters)
            match_counts.append(len(inliers))
            
            if (i + 1) % 10 == 0:
                print(f"Frame {i+1}/{len(test_frames)}: "
                      f"Error={error_meters:.3f}m, "
                      f"Matches={len(inliers)}/{len(matched_3d)}")
                MemoryMonitor.print_memory(f"After {i+1} frames")
        
    print("\n" + "="*60)
    print("CONTINUOUS LOCALIZATION RESULTS")
    print("="*60)
    print(f"Total frames processed: {len(test_frames)}")
    print(f"Successful localizations: {len(errors)}")
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
    
    MemoryMonitor.print_memory("After continuous localization")
    
    print("="*60)