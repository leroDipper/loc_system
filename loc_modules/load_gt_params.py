import json
import numpy as np


class GroundTruthParams:

    @staticmethod
    def load_transformation(json_path):
        """Load the COLMAP to ground truth transformation."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        scale = data['scale']
        R = np.array(data['rotation'])
        t = np.array(data['translation'])
        
        return scale, R, t

    @staticmethod
    def colmap_to_meters(colmap_pos, scale, R, t):
        """Transform COLMAP position to ground truth (meters)."""
        return scale * (R @ colmap_pos) + t

    @staticmethod
    def load_ground_truth(json_path):
        """Load ground truth camera positions."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        gt_poses = {}
        for pose_data in data['poses']:
            frame_num = pose_data['frame']
            frame_name = f"frame_{frame_num:04d}.jpg"
            translation = np.array(pose_data['left_camera']['translation'])
            gt_poses[frame_name] = translation
        
        return gt_poses
