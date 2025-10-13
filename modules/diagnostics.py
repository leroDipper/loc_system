# modules/diagnostics.py
import numpy as np
import cv2

class VODiagnostics:
    def __init__(self, K):
        self.K = K
        self.logs = []

    def analyze_matches(self, frame_idx, kp1, kp2, matches, mask, R, t):
        """Log diagnostics for a given frame in VO."""
        log = {"frame": frame_idx}

        if matches is None or len(matches) == 0:
            self.logs.append(log)
            return

        # Inlier stats
        inliers = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
        num_inliers = np.sum(inliers)
        log["matches"] = len(matches)
        log["inliers"] = int(num_inliers)
        log["inlier_ratio"] = num_inliers / len(matches) if len(matches) > 0 else 0.0

        # Feature displacement (parallax)
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        disp = np.linalg.norm(pts2 - pts1, axis=1)
        log["mean_parallax"] = float(np.mean(disp))
        log["median_parallax"] = float(np.median(disp))

        # Epipolar error instead of reprojection error
        if num_inliers > 0:
            pts1_in = np.float32([kp1[m.queryIdx].pt for i, m in enumerate(matches) if inliers[i]])
            pts2_in = np.float32([kp2[m.trainIdx].pt for i, m in enumerate(matches) if inliers[i]])

            # Normalize using camera intrinsics
            pts1_norm = cv2.undistortPoints(pts1_in.reshape(-1,1,2), self.K, None)
            pts2_norm = cv2.undistortPoints(pts2_in.reshape(-1,1,2), self.K, None)

            # Build skew-symmetric matrix for translation vector
            t = t.reshape(3,1)
            tx = np.array([
                [0, -t[2,0], t[1,0]],
                [t[2,0], 0, -t[0,0]],
                [-t[1,0], t[0,0], 0]
            ])
            E_est = tx @ R  # Essential matrix from R, t

            errors = []
            for p1, p2 in zip(pts1_norm, pts2_norm):
                x1 = np.array([p1[0][0], p1[0][1], 1.0])
                x2 = np.array([p2[0][0], p2[0][1], 1.0])
                err = abs(x2 @ E_est @ x1)
                errors.append(err)

            log["mean_epi_error"] = float(np.mean(errors))
            log["max_epi_error"] = float(np.max(errors))

        # Pose increment magnitudes
        t_norm = np.linalg.norm(t)
        angle = np.degrees(np.arccos(
            np.clip((np.trace(R) - 1) / 2, -1, 1)
        ))
        log["delta_t_norm"] = float(t_norm)
        log["delta_R_angle_deg"] = float(angle)

        self.logs.append(log)

    def get_logs(self):
        """Return logs as a list of dicts (can save to CSV)."""
        return self.logs
