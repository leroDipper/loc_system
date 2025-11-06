"""Map loading functionality."""

import numpy as np
import json


class MapLoader:
    """Handles loading of 3D maps and camera intrinsics."""
    
    @staticmethod
    def load_map(map_path):
        """
        Load 3D map with descriptors from NPZ file.
        
        Args:
            map_path: Path to .npz file containing xyz_world and descriptors
            
        Returns:
            tuple: (xyz_world, descriptors) where:
                - xyz_world: N×3 array of 3D point coordinates
                - descriptors: N×128 array of SIFT descriptors
        """
        data = np.load(map_path)
        xyz_world = data['xyz_world']
        descriptors = data['descriptors']
        
        if len(xyz_world) != len(descriptors):
            raise ValueError("Mismatch between number of 3D points and descriptors")
            
        return xyz_world, descriptors
    
    @staticmethod
    def load_camera_intrinsics(json_path):
        """
        Load camera intrinsics from ground truth JSON file.
        
        Args:
            json_path: Path to JSON file with camera_intrinsics
            
        Returns:
            np.ndarray: 3×3 camera intrinsics matrix K
        """
        with open(json_path, 'r') as f:
            gt = json.load(f)
        
        ci = gt["camera_intrinsics"]
        K = np.array([
            [ci["fx"], 0, ci["cx"]],
            [0, ci["fy"], ci["cy"]],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return K
    
    @staticmethod
    def get_default_intrinsics():
        """
        Get default camera intrinsics for Blender dataset.
        
        Returns:
            np.ndarray: 3×3 default camera intrinsics matrix
        """
        return np.array([
            [1280, 0, 640],
            [0, 1280, 480],
            [0, 0, 1]
        ], dtype=np.float32)