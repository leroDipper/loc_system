from dataclasses import dataclass, field
import numpy as np


@dataclass
class LivePoint:
    xyz: np.ndarray
    descriptor: np.ndarray
    first_kf_id: int
    observations: dict = field(default_factory=dict)
    n_visible: int = 1
    n_found: int = 1
    is_bad: bool = False


class LiveMap:
    def __init__(self):
        self.recent_points = []
        self.good_points = []

    def add_points(self, points: list[LivePoint]):
        self.recent_points.extend(points)

    def get_active_points(self):
        active = []
        for i, point in enumerate(self.recent_points):
            if not point.is_bad:
                active.append((point.xyz, point.descriptor, 'recent', i))
        for i, point in enumerate(self.good_points):
            if not point.is_bad:
                active.append((point.xyz, point.descriptor, 'good', i))

        if not active:
            return None, None, None

        xyz_array  = np.array([a[0] for a in active], dtype=np.float32)
        desc_array = np.array([a[1] for a in active], dtype=np.float32)
        indices    = [(a[2], a[3]) for a in active]

        return xyz_array, desc_array, indices

    def increment_visible(self, indices):
        for idx in indices:
            if idx[0] == 'recent':
                self.recent_points[idx[1]].n_visible += 1
            else:
                self.good_points[idx[1]].n_visible += 1

    def increment_found(self, indices):
        for idx in indices:
            if idx[0] == 'recent':
                self.recent_points[idx[1]].n_found += 1
            else:
                self.good_points[idx[1]].n_found += 1

    def cull(self, current_kf_id):
        surviving = []
        for point in self.recent_points:
            if point.n_found / point.n_visible < 0.25:
                point.is_bad = True
            elif current_kf_id - point.first_kf_id >= 2 and point.n_visible <= 2:
                point.is_bad = True
            elif current_kf_id - point.first_kf_id >= 3:
                self.good_points.append(point)
            else:
                surviving.append(point)
        self.recent_points = surviving

    def add_observation(self, index: tuple, frame_id: int, kp_idx: int):
        source, i = index
        if source == 'recent':
            self.recent_points[i].observations[frame_id] = kp_idx
        else:
            self.good_points[i].observations[frame_id] = kp_idx

    def search_by_projection(self, T, K, image_shape, descriptors, keypoints, n_static_inliers):
        active_xyz, active_desc, indices = self.get_active_points()
        if active_xyz is None:
            return [], [], []

        r = 20 if n_static_inliers >= 20 else 50

        # project all active points into the predicted pose
        Tcw = T[:3, :]
        x3Dh = np.hstack([active_xyz, np.ones((len(active_xyz), 1))])
        x_cam = (Tcw @ x3Dh.T).T

        # discard points behind camera
        valid_mask = x_cam[:, 2] > 0
        valid_idx = np.where(valid_mask)[0]

        # project valid points to pixel coordinates
        x_proj = (K @ x_cam[valid_idx].T).T
        x_proj /= x_proj[:, 2:3]
        u_coords = x_proj[:, 0]
        v_coords = x_proj[:, 1]

        mask = (u_coords >= 0) & (u_coords < image_shape[1]) & (v_coords >= 0) & (v_coords < image_shape[0])

        valid_coord_idx = np.where(mask)[0]

        u_coords = u_coords[valid_coord_idx]
        v_coords = v_coords[valid_coord_idx]

        original_idx = valid_idx[valid_coord_idx]
        visible_indices = [indices[i] for i in original_idx]

        visible_xyz   = [active_xyz[i] for i in original_idx]
        visible_descs = [active_desc[i] for i in original_idx]
        visible_uv    = [(u_coords[j], v_coords[j]) for j in range(len(original_idx))]

        self.increment_visible(visible_indices)

        found_indices = []
        matched_xyz   = []
        matched_uv    = []
        matched_indices = []

        for i in range(len(visible_indices)):
            best_dist = np.inf
            second_best_dist = np.inf
            best_kp_idx = -1

            u_pred, v_pred = visible_uv[i]
            visible_descriptor = visible_descs[i].astype(np.float32)

            for j in range(len(keypoints)):
                diffs = keypoints[j] - np.array([u_pred, v_pred])
                pixel_dist = np.linalg.norm(diffs)

                if pixel_dist < r:
                    dist = visible_descriptor - descriptors[j].astype(np.float32)
                    abs_dist = np.linalg.norm(dist)

                    if abs_dist < best_dist:
                        second_best_dist = best_dist
                        best_dist = abs_dist
                        best_kp_idx = j
                    elif abs_dist < second_best_dist:
                        second_best_dist = abs_dist

            if best_kp_idx >= 0 and best_dist < 0.80 * second_best_dist:
                matched_xyz.append(visible_xyz[i])
                matched_uv.append(keypoints[best_kp_idx])
                matched_indices.append(visible_indices[i])
                found_indices.append(visible_indices[i])

        self.increment_found(found_indices)
        return matched_xyz, matched_uv, matched_indices