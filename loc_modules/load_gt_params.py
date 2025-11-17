import json
import numpy as np
import os
import yaml


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
    
    @staticmethod
    def load_tum_ground_truth(gt_file_path, rgb_file_path):
        """
        Load TUM ground truth poses and associate with image files.
        
        Args:
            gt_file_path: Path to groundtruth.txt
            rgb_file_path: Path to rgb.txt (associates timestamps with image files)
        
        Returns:
            dict: {image_name: position_xyz}
        """
        # Load ground truth poses
        gt_poses = {}
        with open(gt_file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) == 8:
                    timestamp = float(parts[0])
                    tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                    gt_poses[timestamp] = np.array([tx, ty, tz])
        
        # Load RGB associations (timestamp -> image file)
        image_timestamps = {}
        with open(rgb_file_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    timestamp = float(parts[0])
                    image_file = parts[1]
                    image_name = os.path.basename(image_file)
                    image_timestamps[image_name] = timestamp
        
        # Associate images with ground truth
        gt_by_image = {}
        for image_name, img_ts in image_timestamps.items():
            # Find closest ground truth timestamp
            closest_ts = min(gt_poses.keys(), key=lambda ts: abs(ts - img_ts))
            if abs(closest_ts - img_ts) < 0.02:  # Within 20ms
                gt_by_image[image_name] = gt_poses[closest_ts]
        
        print(f"Loaded ground truth for {len(gt_by_image)} images")
        return gt_by_image
    

    @staticmethod
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

