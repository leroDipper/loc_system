"""Main localisation orchestrator."""

import numpy as np
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
                 confidence=0.99):
        """
        Initialize localiser with map and camera parameters.
        
        Args:
            xyz_world: N×3 array of 3D map points
            map_descriptors: N×128 array of map SIFT descriptors
            K: 3×3 camera intrinsics matrix
            ratio_threshold: Lowe's ratio test threshold
            reprojection_error: RANSAC reprojection error (pixels)
            confidence: RANSAC confidence level
        """
        self.xyz_world = np.array(xyz_world, dtype=np.float32)
        self.map_descriptors = np.array(map_descriptors, dtype=np.float32)
        self.K = np.array(K, dtype=np.float32)
        
        # Initialize pipeline components
        self.feature_extractor = FeatureExtractor()
        self.matcher = FeatureMatcher(ratio_threshold=ratio_threshold)
        self.pose_estimator = PoseEstimator(
            reprojection_error=reprojection_error,
            confidence=confidence
        )
    
    def localise(self, image_path):
        """
        localise a query image.
        
        Args:
            image_path: Path to query image
            
        Returns:
            tuple: (result, error_message) where:
                - result: Dict with pose info if successful, None if failed
                - error_message: String describing error, None if successful
        """
        # Extract features
        try:
            keypoints, descriptors = self.feature_extractor.extract(image_path)
        except Exception as e:
            return None, f"Feature extraction failed: {str(e)}"
        
        if descriptors is None or len(descriptors) < 4:
            return None, "Not enough features detected in query image"
        
        # Match features
        matched_map_indices, matched_2d_points = self.matcher.match(
            self.map_descriptors,
            descriptors,
            keypoints
        )
        
        if matched_map_indices is None or len(matched_map_indices) < 4:
            return None, "Not enough feature matches found"
        
        # Get corresponding 3D points
        matched_3d_points = self.xyz_world[matched_map_indices]
        
        # Estimate pose
        pose = self.pose_estimator.estimate_pose(
            matched_3d_points,
            matched_2d_points,
            self.K
        )
        
        if pose is None:
            return None, "Pose estimation failed (RANSAC)"
        
        return pose, None
    
    def localise_batch(self, image_paths):
        """
        localise multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            list: List of (result, error) tuples for each image
        """
        results = []
        for path in image_paths:
            result, error = self.localise(path)
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