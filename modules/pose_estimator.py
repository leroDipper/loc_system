"""Camera pose estimation functionality."""

import cv2
import numpy as np


class PoseEstimator:
    """Estimates camera pose using PnP with RANSAC."""
    
    def __init__(self, reprojection_error=8.0, confidence=0.99):
        """
        Initialize pose estimator.
        
        Args:
            reprojection_error: RANSAC reprojection error threshold (pixels)
            confidence: RANSAC confidence level (0-1)
        """
        self.reprojection_error = reprojection_error
        self.confidence = confidence
        self.dist_coeffs = np.zeros(5, dtype=np.float32)  # No distortion
    
    def estimate_pose(self, points_3d, points_2d, K):
        """
        Estimate camera pose from 2D-3D correspondences.
        
        Args:
            points_3d: N×3 array of 3D world points
            points_2d: N×2 array of corresponding 2D image points
            K: 3×3 camera intrinsics matrix
            
        Returns:
            dict or None: Camera pose information containing:
                - 'position': 3D camera center (x, y, z)
                - 'rotation': 3×3 rotation matrix
                - 'translation': 3D translation vector
                - 'inliers': Number of inlier correspondences
                - 'total': Total number of correspondences
            Returns None if estimation fails
        """
        if len(points_3d) < 4 or len(points_2d) < 4:
            return None
        
        # Ensure correct types
        points_3d = np.array(points_3d, dtype=np.float32)
        points_2d = np.array(points_2d, dtype=np.float32)
        K = np.array(K, dtype=np.float32)
        
        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            self.dist_coeffs,
            reprojectionError=self.reprojection_error,
            confidence=self.confidence
        )
        
        if not success or inliers is None:
            return None
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Compute camera center: C = -R^T @ t
        t = tvec.flatten()
        C = -R.T @ t
        
        return {
            'position': C,
            'rotation': R,
            'translation': t,
            'inliers': len(inliers),
            'total': len(points_3d)
        }
    
    def get_configuration(self):
        """Get current estimator configuration."""
        return {
            'reprojection_error': self.reprojection_error,
            'confidence': self.confidence
        }