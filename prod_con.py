import torch
import os
import cv2
import numpy as np
import time
import glob
import yaml
import queue
import threading
from scipy.spatial.transform import Rotation

from accelerated_modules import vocab_tree_match
from loc_modules.load_gt_params import GroundTruthParams
from test.memory import MemoryMonitor
from imu.euroc_imu import EurocIMU
from imu.ekf import ErrorStateEKF

QUEUE_MAXSIZE = 3
SENTINEL     = None


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


# ── Thread 1: Reader ──────────────────────────────────────────────────────────

def reader_thread(test_image_list, gt_poses, q_out):
    """
    Reads images from disk and pushes (frame_name, timestamp, frame_gray) onto q_out.
    Skips frames with no ground truth or unreadable images.
    """
    for frame_path in test_image_list:
        frame_name     = os.path.basename(frame_path)
        curr_timestamp = int(frame_name.replace('.png', ''))

        if frame_name not in gt_poses:
            continue

        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q_out.put((frame_name, curr_timestamp, frame_gray))

    q_out.put(SENTINEL)


# ── Thread 2: Extractor ───────────────────────────────────────────────────────

def extractor_thread(q_in, q_out, timings):
    """
    Owns the XFeat model. Takes frames from q_in, runs inference,
    pushes (frame_name, timestamp, keypoints, descriptors) onto q_out.
    """
    torch.set_num_threads(4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xfeat  = torch.hub.load('verlab/accelerated_features', 'XFeat',
                            pretrained=True, top_k=4096)
    xfeat  = xfeat.to(device).eval()

    while True:
        item = q_in.get()
        if item is SENTINEL:
            q_out.put(SENTINEL)
            break

        frame_name, timestamp, frame_gray = item

        t_start = time.time()
        with torch.no_grad():
            output = xfeat.detectAndCompute(frame_gray, top_k=250)
        timings['extract'].append(time.time() - t_start)

        feats       = output[0]
        keypoints   = feats['keypoints'].cpu().numpy()
        descriptors = feats['descriptors'].cpu().numpy()
        descriptors = np.clip((descriptors + 0.5) * 255.0, 0, 255).astype(np.uint8)

        q_out.put((frame_name, timestamp, keypoints, descriptors))


# ── Thread 3: Matcher ─────────────────────────────────────────────────────────

def matcher_thread(q_in, q_out, matcher, map_3d_points, timings):
    """
    Takes (frame_name, timestamp, keypoints, descriptors) from q_in.
    Runs vocab tree match, pushes (frame_name, timestamp, matched_2d, matched_3d) onto q_out.
    Drops frames with too few matches rather than propagating them.
    """
    while True:
        item = q_in.get()
        if item is SENTINEL:
            q_out.put(SENTINEL)
            break

        frame_name, timestamp, keypoints, descriptors = item

        t_start = time.time()
        query_idx, map_idx, distances = matcher.match(descriptors, ratio_threshold=0.80)
        timings['match'].append(time.time() - t_start)

        query_to_map = {}
        for q_idx, m_idx, dist in zip(query_idx, map_idx, distances):
            if q_idx not in query_to_map or dist < query_to_map[q_idx][1]:
                query_to_map[q_idx] = (m_idx, dist)

        if len(query_to_map) < 4:
            continue

        matched_3d = np.array([map_3d_points[m] for _, (m, _) in query_to_map.items()],
                               dtype=np.float32)
        matched_2d = np.array([keypoints[q] for q in query_to_map.keys()],
                               dtype=np.float32)

        q_out.put((frame_name, timestamp, matched_2d, matched_3d))


# ── Thread 4: PnP + EKF ───────────────────────────────────────────────────────

def pnp_ekf_thread(q_in, imu, ekf, gt_poses, K_cv, dist_coeffs,
                   scale, R_gt, t_gt, results, timings):
    """
    Strictly sequential — processes frames in queue order to preserve the
    IMU temporal chain. Runs IMU prediction, PnP, reprojection check, EKF update.
    """
    ekf_initialised = False
    prev_timestamp  = None

    R_CB = imu.T_CB[:3, :3]
    t_CB = imu.T_CB[:3, 3]
    R_BC = R_CB.T
    t_BC = -R_CB.T @ t_CB

    while True:
        item = q_in.get()
        if item is SENTINEL:
            break

        frame_name, curr_timestamp, matched_2d, matched_3d = item

        # ── IMU prediction ────────────────────────────────────────────
        if prev_timestamp is not None and ekf_initialised:
            readings = imu.get_readings_between(prev_timestamp, curr_timestamp)
            for reading in readings:
                ekf.predict(reading)

        # ── PnP ───────────────────────────────────────────────────────
        t_start = time.time()
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            matched_3d, matched_2d, K_cv, dist_coeffs,
            reprojectionError=8.0, confidence=0.99
        )
        timings['pnp'].append(time.time() - t_start)

        if not success or inliers is None:
            prev_timestamp = curr_timestamp
            results['pnp_failed'] += 1
            continue

        if len(inliers) < 6:
            prev_timestamp = curr_timestamp
            results['rejected_low_inliers'] += 1
            continue

        # ── Reprojection error check ──────────────────────────────────
        inlier_3d = matched_3d[inliers.flatten()]
        inlier_2d = matched_2d[inliers.flatten()]
        projected, _ = cv2.projectPoints(inlier_3d, rvec, tvec, K_cv, dist_coeffs)
        median_reproj = np.median(np.linalg.norm(inlier_2d - projected.squeeze(), axis=1))
        if median_reproj > 5.0:
            prev_timestamp = curr_timestamp
            results['rejected_reproj_error'] += 1
            continue

        # ── Pose ──────────────────────────────────────────────────────
        R_cam    = cv2.Rodrigues(rvec)[0]
        C_colmap = -R_cam.T @ tvec.flatten()
        C_metres = GroundTruthParams.colmap_to_meters(C_colmap, scale, R_gt, t_gt)
        C_gt     = gt_poses[frame_name]

        pnp_error = np.linalg.norm(C_metres - C_gt)
        results['pnp_errors'].append(pnp_error)
        results['match_counts'].append(len(inliers))

        # ── EKF ───────────────────────────────────────────────────────
        if not ekf_initialised:
            ekf.initialise(C_metres, R_cam)
            ekf_initialised = True
            prev_timestamp  = curr_timestamp
            results['ekf_errors'].append(pnp_error)
            continue

        ekf.update(C_metres, R_cam)
        fused_p, _ = ekf.get_state()

        fused_cam_pos = R_BC @ fused_p + t_BC
        ekf_error     = np.linalg.norm(fused_cam_pos - C_gt)
        results['ekf_errors'].append(ekf_error)

        prev_timestamp = curr_timestamp


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    EUROC_DATASET_PATH = 'resources/mh_01'
    CAMERA_PARAMS_PATH = 'resources/mh_01/images/camera_rectified.yaml'
    N_TRAIN_IMAGES     = 2900
    test_dataset_path  = 'resources/mh_01/images'

    scale, R_gt, t_gt = GroundTruthParams.load_transformation(
        'resources/mh_01/project_files/colmap_to_gt_transform.json')

    gt_poses = GroundTruthParams.load_euroc_ground_truth_by_image(
        gt_csv_path=os.path.join(EUROC_DATASET_PATH, 'data.csv'),
        image_dir=os.path.join(EUROC_DATASET_PATH, 'images')
    )

    vocabulary = 'resources/mh_01/vocabularies/vocab_tree_master.bin'
    data = np.load('resources/mh_01/map_databases/mh_01_master.npz')
    map_3d_points   = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    MemoryMonitor.print_memory("After loading map")

    camera_params = load_camera_params(CAMERA_PARAMS_PATH)
    K_cv = np.array([
        [camera_params['fx'], 0,                   camera_params['cx']],
        [0,                   camera_params['fy'],  camera_params['cy']],
        [0,                   0,                    1]
    ], dtype=np.float32)
    dist_coeffs = np.zeros(5, dtype=np.float32)

    print("Building vocabulary matcher...")
    matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)

    MemoryMonitor.print_memory("After building matcher")

    print("Loading IMU data...")
    imu = EurocIMU(EUROC_DATASET_PATH)
    ekf = ErrorStateEKF(imu.Q, imu.R, imu.T_CB)
    print("IMU and EKF initialised")

    colmap_images        = load_colmap_image_names('resources/mh_01/project_files/images.txt')
    colmap_set           = set(colmap_images)
    all_frames           = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    all_filenames        = [os.path.basename(f) for f in all_frames]
    reconstructed_chrono = [f for f in all_filenames if f in colmap_set]
    test_images          = [os.path.join(test_dataset_path, f)
                            for f in reconstructed_chrono[N_TRAIN_IMAGES:]]
    test_image_list      = sorted(test_images)

    print(f"Map built with {N_TRAIN_IMAGES} images")
    print(f"Testing with {len(test_image_list)} held-out images\n")

    # ── Shared state ──────────────────────────────────────────────────
    timings = {'extract': [], 'match': [], 'pnp': []}
    results = {
        'pnp_errors':            [],
        'ekf_errors':            [],
        'match_counts':          [],
        'pnp_failed':            0,
        'rejected_low_inliers':  0,
        'rejected_reproj_error': 0,
    }

    # ── Queues ────────────────────────────────────────────────────────
    q_read    = queue.Queue(maxsize=QUEUE_MAXSIZE)
    q_extract = queue.Queue(maxsize=QUEUE_MAXSIZE)
    q_match   = queue.Queue(maxsize=QUEUE_MAXSIZE)

    # ── Launch threads ────────────────────────────────────────────────
    t_wall_start = time.time()

    threads = [
        threading.Thread(target=reader_thread,
                         args=(test_image_list, gt_poses, q_read),
                         daemon=True),

        threading.Thread(target=extractor_thread,
                         args=(q_read, q_extract, timings),
                         daemon=True),

        threading.Thread(target=matcher_thread,
                         args=(q_extract, q_match, matcher, map_3d_points, timings),
                         daemon=True),

        threading.Thread(target=pnp_ekf_thread,
                         args=(q_match, imu, ekf, gt_poses, K_cv, dist_coeffs,
                               scale, R_gt, t_gt, results, timings),
                         daemon=True),
    ]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    t_wall_total = time.time() - t_wall_start

    # ── Results ───────────────────────────────────────────────────────
    pnp_errors  = results['pnp_errors']
    ekf_errors  = results['ekf_errors']

    print("\n" + "="*60)
    print("LOCALISATION ACCURACY — RAW PnP vs EKF FUSED")
    print("="*60)
    print(f"Total test frames:         {len(test_image_list)}")
    print(f"PnP failed:                {results['pnp_failed']}")
    print(f"Rejected (low inliers):    {results['rejected_low_inliers']}")
    print(f"Rejected (high reproj):    {results['rejected_reproj_error']}")
    print(f"Successful localisations:  {len(pnp_errors)}")
    if len(test_image_list) > 0:
        print(f"Success rate:              {len(pnp_errors)/len(test_image_list)*100:.1f}%")

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
        sequential_total = np.mean(timings['extract']) + np.mean(timings['match']) + np.mean(timings['pnp'])
        print(f"  Sequential total:   {sequential_total*1000:.2f} ms  ({1.0/sequential_total:.2f} FPS)")

    print(f"\n  Wall-clock total:   {t_wall_total:.2f} s")
    if len(test_image_list) > 0:
        print(f"  Wall-clock FPS:     {len(test_image_list)/t_wall_total:.2f}")

    MemoryMonitor.print_memory("After localisation")
    print("="*60)