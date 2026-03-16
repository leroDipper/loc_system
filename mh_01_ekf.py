import torch
import os
import cv2
import numpy as np
import time
import glob
import yaml
from scipy.spatial.transform import Rotation

from accelerated_modules import vocab_tree_match
from loc_modules.load_gt_params import GroundTruthParams
from test.memory import MemoryMonitor
from imu.euroc_imu import EurocIMU
from imu.ekf import ErrorStateEKF


def load_camera_params(yaml_path):
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    return {
        'width':  params['resolution'][0],
        'height': params['resolution'][1],
        'fx': params['intrinsics'][0],
        'fy': params['intrinsics'][1],
        'cx': params['intrinsics'][2],
        'cy': params['intrinsics'][3]
    }


def load_colmap_image_names(images_txt_path):
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

    EUROC_DATASET_PATH = 'resources/mh_01'
    CAMERA_PARAMS_PATH = 'resources/mh_01/images/camera_rectified.yaml'
    N_TRAIN_IMAGES     = 2900
    test_dataset_path  = 'resources/mh_01/images'

    # ── Ground truth ──────────────────────────────────────────────────
    scale, R_gt, t_gt = GroundTruthParams.load_transformation(
        'resources/mh_01/project_files/colmap_to_gt_transform.json')

    gt_poses = GroundTruthParams.load_euroc_ground_truth_by_image(
        gt_csv_path=os.path.join(EUROC_DATASET_PATH, 'data.csv'),
        image_dir=os.path.join(EUROC_DATASET_PATH, 'images')
    )

    # ── XFeat model ───────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat',
                               pretrained=True, top_k=4096)
        xfeat = xfeat.to(device).eval()
    except Exception as e:
        print(f"Error loading XFeat: {e}")
        exit()

    MemoryMonitor.print_memory("After loading XFeat")

    # ── Map and vocabulary ────────────────────────────────────────────
    vocabulary = 'resources/mh_01/vocabularies/vocab_tree_master.bin'
    data = np.load('resources/mh_01/map_databases/mh_01_master.npz')
    map_3d_points   = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

    # ── Camera ────────────────────────────────────────────────────────
    camera_params = load_camera_params(CAMERA_PARAMS_PATH)
    K_cv = np.array([
        [camera_params['fx'], 0,                   camera_params['cx']],
        [0,                   camera_params['fy'],  camera_params['cy']],
        [0,                   0,                    1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    # ── Vocab tree matcher ────────────────────────────────────────────
    print("Building vocabulary matcher...")
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)

    MemoryMonitor.print_memory("After building matcher")

    # ── IMU + EKF ─────────────────────────────────────────────────────
    print("Loading IMU data...")
    imu = EurocIMU(EUROC_DATASET_PATH)
    ekf = ErrorStateEKF(imu.Q, imu.R, imu.T_CB)
    print("IMU and EKF initialised")

    # ── Test images ───────────────────────────────────────────────────
    colmap_images        = load_colmap_image_names('resources/mh_01/project_files/images.txt')
    colmap_set           = set(colmap_images)
    all_frames           = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    all_filenames        = [os.path.basename(f) for f in all_frames]
    reconstructed_chrono = [f for f in all_filenames if f in colmap_set]
    test_images          = [os.path.join(test_dataset_path, f)
                            for f in reconstructed_chrono[N_TRAIN_IMAGES:]]

    print(f"\nMap built with {N_TRAIN_IMAGES} images")
    print(f"Testing with {len(test_images)} held-out images\n")

    # ── Tracking ──────────────────────────────────────────────────────
    pnp_errors   = []
    ekf_errors   = []
    match_counts = []
    timings      = {'extract': [], 'match': [], 'pnp': []}

    skipped_no_gt           = 0
    skipped_no_image        = 0
    skipped_too_few_matches = 0
    pnp_failed              = 0
    rejected_low_inliers    = 0
    rejected_reproj_error   = 0

    ekf_initialised = False
    prev_timestamp  = None
    test_image_list = sorted(test_images)

    # ── Main loop ─────────────────────────────────────────────────────
    for i, frame_path in enumerate(test_image_list):
        frame_name     = os.path.basename(frame_path)
        curr_timestamp = int(frame_name.replace('.png', ''))

        if frame_name not in gt_poses:
            skipped_no_gt += 1
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            skipped_no_image += 1
            continue
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── IMU prediction ────────────────────────────────────────────
        if prev_timestamp is not None and ekf_initialised:
            readings = imu.get_readings_between(prev_timestamp, curr_timestamp)
            for reading in readings:
                ekf.predict(reading)

        # ── Feature extraction ────────────────────────────────────────
        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=250)
        timings['extract'].append(time.time() - t_start)

        feats       = output[0]
        keypoints   = feats['keypoints'].cpu().numpy()
        descriptors = feats['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)

        # ── Matching ──────────────────────────────────────────────────
        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80)
        timings['match'].append(time.time() - t_start)

        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)

        matched_3d = np.array([map_3d_points[m] for _, (m, _) in query_to_map.items()],
                               dtype=np.float32)
        matched_2d = np.array([keypoints[q] for q in query_to_map.keys()],
                               dtype=np.float32)

        if len(matched_3d) < 4:
            skipped_too_few_matches += 1
            prev_timestamp = curr_timestamp
            continue

        # ── PnP ───────────────────────────────────────────────────────
        t_start = time.time()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_3d, matched_2d, K_cv, dist_coeffs,
            reprojectionError=8.0, confidence=0.99
        )
        timings['pnp'].append(time.time() - t_start)

        if not success or inliers is None:
            pnp_failed += 1
            prev_timestamp = curr_timestamp
            continue

        if len(inliers) < 6:
            rejected_low_inliers += 1
            prev_timestamp = curr_timestamp
            continue

        # ── Reprojection error check ──────────────────────────────────
        inlier_points_3d = matched_3d[inliers.flatten()]
        inlier_points_2d = matched_2d[inliers.flatten()]
        projected, _ = cv2.projectPoints(inlier_points_3d, rvec, tvec, K_cv, dist_coeffs)
        median_reproj_error = np.median(np.linalg.norm(inlier_points_2d - projected.squeeze(), axis=1))
        if median_reproj_error > 5.0:
            rejected_reproj_error += 1
            prev_timestamp = curr_timestamp
            continue

        # ── Pose from PnP ─────────────────────────────────────────────
        R_cam    = cv2.Rodrigues(rvec)[0]
        C_colmap = -R_cam.T @ tvec.flatten()
        C_metres = GroundTruthParams.colmap_to_meters(C_colmap, scale, R_gt, t_gt)
        C_gt     = gt_poses[frame_name]

        pnp_error = np.linalg.norm(C_metres - C_gt)
        pnp_errors.append(pnp_error)
        match_counts.append(len(inliers))

        # ── EKF initialisation from first good pose ───────────────────
        if not ekf_initialised:
            ekf.initialise(C_metres, R_cam)
            ekf_initialised = True
            prev_timestamp  = curr_timestamp
            ekf_errors.append(pnp_error)
            continue

        # ── EKF update ────────────────────────────────────────────────
        ekf.update(C_metres, R_cam)
        fused_p, _ = ekf.get_state()

        # convert EKF body-frame position back to camera frame for evaluation
        R_CB        = imu.T_CB[:3, :3]
        t_CB        = imu.T_CB[:3, 3]
        R_BC        = R_CB.T
        t_BC        = -R_CB.T @ t_CB
        fused_cam_pos = R_BC @ fused_p + t_BC

        ekf_error = np.linalg.norm(fused_cam_pos - C_gt)
        ekf_errors.append(ekf_error)

        prev_timestamp = curr_timestamp

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("LOCALISATION ACCURACY — RAW PnP vs EKF FUSED")
    print("="*60)
    print(f"Total test frames:         {len(test_image_list)}")
    print(f"Skipped (no GT):           {skipped_no_gt}")
    print(f"Skipped (no image):        {skipped_no_image}")
    print(f"Skipped (too few matches): {skipped_too_few_matches}")
    print(f"PnP failed:                {pnp_failed}")
    print(f"Rejected (low inliers):    {rejected_low_inliers}")
    print(f"Rejected (high reproj):    {rejected_reproj_error}")
    print(f"Successful localisations:  {len(pnp_errors)}")
    print(f"Success rate:              {len(pnp_errors)/(len(test_image_list)-skipped_no_gt)*100:.1f}%")

    if len(pnp_errors) > 0:
        print("\nRaw PnP:")
        print(f"  Mean error:   {np.mean(pnp_errors):.4f} m  ({np.mean(pnp_errors)*100:.2f} cm)")
        print(f"  Median error: {np.median(pnp_errors):.4f} m  ({np.median(pnp_errors)*100:.2f} cm)")
        print(f"  Std:          {np.std(pnp_errors):.4f} m")
        print(f"  Max:          {np.max(pnp_errors):.4f} m")
        below_20 = np.sum(np.array(pnp_errors) <= 0.20)
        print(f"  Within 20cm:  {below_20}/{len(pnp_errors)} ({below_20/len(pnp_errors)*100:.1f}%)")

    if len(ekf_errors) > 0:
        print("\nEKF Fused:")
        print(f"  Mean error:   {np.mean(ekf_errors):.4f} m  ({np.mean(ekf_errors)*100:.2f} cm)")
        print(f"  Median error: {np.median(ekf_errors):.4f} m  ({np.median(ekf_errors)*100:.2f} cm)")
        print(f"  Std:          {np.std(ekf_errors):.4f} m")
        print(f"  Max:          {np.max(ekf_errors):.4f} m")
        below_20 = np.sum(np.array(ekf_errors) <= 0.20)
        print(f"  Within 20cm:  {below_20}/{len(ekf_errors)} ({below_20/len(ekf_errors)*100:.1f}%)")

    if len(pnp_errors) > 0 and len(ekf_errors) > 0:
        improvement = (np.mean(pnp_errors) - np.mean(ekf_errors)) / np.mean(pnp_errors) * 100
        print(f"\nMean error improvement: {improvement:.1f}%")

    if len(timings['extract']) > 0:
        print(f"\nTiming (per frame):")
        print(f"  Feature extraction: {np.mean(timings['extract'])*1000:.2f} ms")
        print(f"  Matching:           {np.mean(timings['match'])*1000:.2f} ms")
        print(f"  PnP:                {np.mean(timings['pnp'])*1000:.2f} ms")
        total = np.mean(timings['extract']) + np.mean(timings['match']) + np.mean(timings['pnp'])
        print(f"  Total:              {total*1000:.2f} ms  ({1.0/total:.2f} FPS)")

    MemoryMonitor.print_memory("After localisation")
    print("="*60)