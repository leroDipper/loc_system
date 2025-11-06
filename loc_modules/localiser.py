"""Main localisation orchestrator."""

import numpy as np
import time
from .feature_extractor import FeatureExtractor
from .matcher import FeatureMatcher
from .pose_estimator import PoseEstimator


class Localiser:
    """
    Main localisation class that orchestrates the full pipeline.
    
    This class coordinates feature extraction, matching, and pose estimation
    to localise a query image against a pre-built 3D map.
    """
    
    def __init__(self, xyz_world, map_descriptors, K, 
                 ratio_threshold=0.75, 
                 reprojection_error=8.0,
                 confidence=0.99,
                 min_inliers=15):
        """
        Initialize localiser with map and camera parameters.
        
        Args:
            xyz_world: N×3 array of 3D map points
            map_descriptors: N×128 array of map SIFT descriptors
            K: 3×3 camera intrinsics matrix
            ratio_threshold: Lowe's ratio test threshold
            reprojection_error: RANSAC reprojection error (pixels)
            confidence: RANSAC confidence level
            min_inliers: Minimum inliers required for valid pose
        """
        self.xyz_world = np.array(xyz_world, dtype=np.float32)
        self.map_descriptors = np.array(map_descriptors, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        
        # Initialize pipeline components
        self.feature_extractor = FeatureExtractor()
        self.matcher = FeatureMatcher(ratio_threshold=ratio_threshold)
        self.pose_estimator = PoseEstimator(
            reprojection_error=reprojection_error,
            confidence=confidence,
            min_inliers=min_inliers
        )
        
        # Set map bounds for outlier detection
        self.pose_estimator.set_map_bounds(self.xyz_world)
    
    def localise(self, image_path, option=None, resize_scale=0.5):
        """
        localise a query image.
        
        Args:
            image_path: Path to query image
            option: 'array', 'resize', or None
            resize_scale: Scale factor when option='resize' (default 0.5 for 640x480)
            
        Returns:
            tuple: (result, error_message) where:
                - result: Dict with pose info if successful, None if failed
                - error_message: String describing error, None if successful
        """
        # Initialize timing dict
        timing = {'extraction': 0, 'matching': 0, 'pnp': 0}
        
        # Extract features
        try:
            t_start = time.time()
            if option == 'array':
                keypoints, descriptors = self.feature_extractor.extract_from_array(image_path)
                K_adjusted = self.K
            elif option == 'resize':
                keypoints, descriptors = self.feature_extractor.resize_and_extract(image_path)
                # Scale K for resized image
                K_adjusted = self.K.copy()
                K_adjusted[0, 0] *= resize_scale  # fx
                K_adjusted[1, 1] *= resize_scale  # fy
                K_adjusted[0, 2] *= resize_scale  # cx
                K_adjusted[1, 2] *= resize_scale  # cy
            else:
                keypoints, descriptors = self.feature_extractor.extract(image_path)
                K_adjusted = self.K
            timing['extraction'] = time.time() - t_start
        except Exception as e:
            return None, f"Feature extraction failed: {str(e)}"
        
        if descriptors is None or len(descriptors) < 4:
            return None, "Not enough features detected in query image"
        
        # Match features
        t_start = time.time()
        matched_map_indices, matched_2d_points = self.matcher.match(
            self.map_descriptors,
            descriptors,
            keypoints
        )
        timing['matching'] = time.time() - t_start
        
        if matched_map_indices is None or len(matched_map_indices) < 4:
            return None, "Not enough feature matches found"
        
        # Get corresponding 3D points
        matched_3d_points = self.xyz_world[matched_map_indices]
        
        # Estimate pose with adjusted K
        t_start = time.time()
        pose = self.pose_estimator.estimate_pose(
            matched_3d_points,
            matched_2d_points,
            K_adjusted
        )
        timing['pnp'] = time.time() - t_start
        
        if pose is None:
            return None, "Pose estimation failed (RANSAC or outlier rejection)"
        
        # Add timing to result
        pose['timing'] = timing
        
        return pose, None
    
    def localise_batch(self, image_paths, option=None, resize_scale=0.5):
        """
        localise multiple images.
        
        Args:
            image_paths: List of image paths
            option: 'array', 'resize', or None
            resize_scale: Scale factor when option='resize'
            
        Returns:
            list: List of (result, error) tuples for each image
        """
        results = []
        for path in image_paths:
            result, error = self.localise(path, option=option, resize_scale=resize_scale)
            results.append((result, error))
        return results
    
    def get_map_info(self):
        """Get information about the loaded map."""
        return {
            'num_points': len(self.xyz_world),
            'descriptor_dim': self.map_descriptors.shape[1],
            'bounds': {
                'min': self.xyz_world.min(axis=0).tolist(),
                'max': self.xyz_world.max(axis=0).tolist()
            }
        }