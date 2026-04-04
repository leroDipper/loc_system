from .live_map import LiveMap, LivePoint
from dataclasses import dataclass
import numpy as np
import cv2
from scipy.optimize import least_squares


@dataclass
class Keyframe:
    frame_id: int
    T: np.ndarray
    keypoints: np.ndarray
    descriptors: np.ndarray
    K: np.ndarray


class LocalMapper:
    def __init__(self, live_map: LiveMap, K: np.ndarray):
        self.live_map = live_map
        self.K = K
        self.keyframes = []

    def add_keyframe(self, frame_id, T, keypoints, descriptors):
        new_kf = Keyframe(frame_id, T, keypoints, descriptors, self.K)
        self.keyframes.append(new_kf)

    def process(self):
        if len(self.keyframes) < 2:
            return

        kf1, kf2 = self.keyframes[-2], self.keyframes[-1]
        matches = self._match_keyframes(kf1, kf2, set())

        for idx1, idx2 in matches:
            kp1 = kf1.keypoints[idx1]
            kp2 = kf2.keypoints[idx2]
            xn1 = np.linalg.inv(kf1.K) @ np.array([kp1[0], kp1[1], 1.0])
            xn2 = np.linalg.inv(kf2.K) @ np.array([kp2[0], kp2[1], 1.0])
            x3D = self._triangulate(xn1, xn2, kf1.T[:3, :], kf2.T[:3, :])
            if x3D is not None and self._check_quality(x3D, xn1, xn2, kf1.T[:3, :], kf2.T[:3, :], kp1, kp2):
                point = LivePoint(
                            xyz=x3D,
                            descriptor=kf2.descriptors[idx2],
                            first_kf_id=kf2.frame_id,
                            observations={kf1.frame_id: idx1, kf2.frame_id: idx2}
                        )
                self.live_map.add_points([point])

        self.live_map.cull(kf2.frame_id)
        self._run_ba()

    def _match_keyframes(self, kf1, kf2, matched_indices_kf1: set):
        F12 = self._compute_fundamental(kf1, kf2)

        d1 = kf1.descriptors.astype(np.float32)
        d2 = kf2.descriptors.astype(np.float32)
        dists = np.linalg.norm(d1[:, None] - d2[None, :], axis=2)

        matches = []
        matched_in_kf2 = set()

        for idx1 in range(len(kf1.keypoints)):
            if idx1 in matched_indices_kf1:
                continue

            kp1 = kf1.keypoints[idx1]

            # epipolar line in image 2
            l2 = F12 @ np.array([kp1[0], kp1[1], 1.0])

            best_dist = np.inf
            second_best_dist = np.inf
            best_idx2 = -1

            for idx2 in range(len(kf2.keypoints)):
                if idx2 in matched_in_kf2:
                    continue

                kp2 = kf2.keypoints[idx2]

                # epipolar distance check
                num = (l2[0]*kp2[0] + l2[1]*kp2[1] + l2[2])**2
                den = l2[0]**2 + l2[1]**2
                if den < 1e-10:
                    continue
                if num/den > 3.84:
                    continue

                d = dists[idx1, idx2]
                if d < best_dist:
                    second_best_dist = best_dist
                    best_dist = d
                    best_idx2 = idx2
                elif d < second_best_dist:
                    second_best_dist = d

            if best_idx2 >= 0 and best_dist < 0.80 * second_best_dist:
                matches.append((idx1, best_idx2))
                matched_in_kf2.add(best_idx2)

        return matches

    def _compute_fundamental(self, kf1, kf2):
        R1w = kf1.T[:3, :3]
        t1w = kf1.T[:3,  3]
        R2w = kf2.T[:3, :3]
        t2w = kf2.T[:3,  3]

        R12 = R1w @ R2w.T
        t12 = -R12 @ t2w + t1w

        t12x = np.array([
            [ 0,      -t12[2],  t12[1]],
            [ t12[2],  0,      -t12[0]],
            [-t12[1],  t12[0],  0     ]
        ])

        K_inv = np.linalg.inv(self.K)
        return K_inv.T @ t12x @ R12 @ K_inv

    def _triangulate(self, xn1, xn2, Tc1w, Tc2w):
        A = np.array([
            xn1[0] * Tc1w[2] - Tc1w[0],
            xn1[1] * Tc1w[2] - Tc1w[1],
            xn2[0] * Tc2w[2] - Tc2w[0],
            xn2[1] * Tc2w[2] - Tc2w[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        x3Dh = Vt[-1]
        if abs(x3Dh[3]) < 1e-7:
            return None
        return x3Dh[:3] / x3Dh[3]

    def _check_quality(self, x3D, xn1, xn2, Tc1w, Tc2w, kp1, kp2):
        ray1 = Tc1w[:3, :3].T @ xn1
        ray2 = Tc2w[:3, :3].T @ xn2
        cos_parallax = ray1.dot(ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
        if cos_parallax > 0.9998:
            return False

        x3Dh = np.hstack((x3D, 1.0))
        z1 = (Tc1w @ x3Dh)[2]
        z2 = (Tc2w @ x3Dh)[2]
        if z1 <= 0 or z2 <= 0:
            return False

        reproj1 = self.K @ (Tc1w @ x3Dh)
        reproj2 = self.K @ (Tc2w @ x3Dh)
        reproj1 /= reproj1[2]
        reproj2 /= reproj2[2]

        error1 = np.linalg.norm(reproj1[:2] - kp1)
        error2 = np.linalg.norm(reproj2[:2] - kp2)
        return error1**2 < 5.991 and error2**2 < 5.991

    def _run_ba(self):
        if len(self.keyframes) < 2:
            return

        window = self.keyframes[-5:]
        anchor = window[0]       # fixed, not optimised
        opt_kfs = window[1:]     # these poses get optimised
        opt_kf_ids = {kf.frame_id for kf in opt_kfs}
        window_ids = {kf.frame_id for kf in window}

        all_points = self.live_map.recent_points + self.live_map.good_points
        ba_points = [
            p for p in all_points if not p.is_bad and len(set(p.observations.keys()) & window_ids) >= 2 
        ]

        if not ba_points:
            return
        
        params = []
        for kf in opt_kfs:
            R = kf.T[:3, :3]
            t = kf.T[:3, 3]
            rvec, _ = cv2.Rodrigues(R)
            params.extend(rvec.flatten())
            params.extend(t.flatten())

        for p in ba_points:
            params.extend(p.xyz)

        params = np.array(params, dtype=np.float64)

        def residuals(x):
            res = []

            # unpack keyframe poses from x
            kf_poses = {}
            kf_poses[anchor.frame_id] = anchor.T[:3, :]  # anchor is fixed

            for i, kf in enumerate(opt_kfs):
                offset = i * 6
                rvec = x[offset:offset+3]
                tvec = x[offset+3:offset+6]
                R, _ = cv2.Rodrigues(rvec)
                Tcw = np.hstack([R, tvec.reshape(3, 1)])
                kf_poses[kf.frame_id] = Tcw

            # unpack point positions from x
            pt_offset = len(opt_kfs) * 6
            for j, point in enumerate(ba_points):
                xyz = x[pt_offset + j*3 : pt_offset + j*3 + 3]

                for frame_id, kp_idx in point.observations.items():
                    if frame_id not in kf_poses:
                        continue

                    # find the keyframe to get the actual keypoint
                    kf = next((k for k in window if k.frame_id == frame_id), None)
                    if kf is None:
                        continue

                    Tcw = kf_poses[frame_id]
                    x3Dh = np.append(xyz, 1.0)

                    # project
                    x_cam = Tcw @ x3Dh
                    if x_cam[2] <= 0:
                        res.extend([100.0, 100.0])  # large penalty for behind camera
                        continue

                    proj = self.K @ x_cam
                    u = proj[0] / proj[2]
                    v = proj[1] / proj[2]

                    kp = kf.keypoints[kp_idx]
                    res.extend([u - kp[0], v - kp[1]])

            return np.array(res, dtype=np.float64)
        
        result = least_squares(residuals, params, method='lm')

        # unpack optimised poses back into keyframes
        for i, kf in enumerate(opt_kfs):
            offset = i * 6
            rvec = result.x[offset:offset+3]
            tvec = result.x[offset+3:offset+6]
            R, _ = cv2.Rodrigues(rvec)
            kf.T[:3, :3] = R
            kf.T[:3,  3] = tvec

        # unpack optimised point positions back into live points
        pt_offset = len(opt_kfs) * 6
        for j, point in enumerate(ba_points):
            point.xyz = result.x[pt_offset + j*3 : pt_offset + j*3 + 3].astype(np.float32)


        
