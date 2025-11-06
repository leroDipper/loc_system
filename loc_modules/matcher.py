"""Feature matching functionality."""

import cv2
import numpy as np


class FeatureMatcher:
    """Matches features between query image and map."""
    
    def __init__(self, ratio_threshold=0.75):
        """
        Initialize feature matcher.
        
        Args:
            ratio_threshold: Lowe's ratio test threshold (default: 0.75)
        """
        self.matcher = cv2.BFMatcher()
        self.ratio_threshold = ratio_threshold
    
    def match(self, map_descriptors, query_descriptors, query_keypoints):
        """
        Match query descriptors to map descriptors.
        
        Args:
            map_descriptors: N×D array of map descriptors (D=64 for XFeat, 128 for SIFT)
            query_descriptors: M×D array of query descriptors
            query_keypoints: N×2 array of query keypoint coordinates (NOT cv2.KeyPoint objects for XFeat)
            
        Returns:
            tuple: (matched_3d_indices, matched_2d_points) where:
                - matched_3d_indices: List of map point indices
                - matched_2d_points: N×2 array of corresponding 2D points
        """
        # Ensure both descriptors have same dtype
        map_descriptors = np.array(map_descriptors, dtype=query_descriptors.dtype)
        query_descriptors = np.array(query_descriptors, dtype=query_descriptors.dtype)
        
        # KNN match
        matches = self.matcher.knnMatch(map_descriptors, query_descriptors, k=2)
        
        # Ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < self.ratio_threshold * n.distance:
                good_matches.append(m)
        
        if len(good_matches) < 4:
            return None, None
        
        # Keep only best match per query keypoint (avoid duplicates)
        query_to_map = {}
        for match in good_matches:
            query_idx = match.trainIdx
            map_idx = match.queryIdx
            distance = match.distance
            
            if query_idx not in query_to_map or distance < query_to_map[query_idx][1]:
                query_to_map[query_idx] = (map_idx, distance)
        
        # Extract indices and points
        matched_map_indices = []
        matched_2d_points = []
        
        for query_idx, (map_idx, _) in query_to_map.items():
            matched_map_indices.append(map_idx)
            # Handle both cv2.KeyPoint objects and numpy arrays
            if hasattr(query_keypoints[query_idx], 'pt'):
                # SIFT keypoint object
                matched_2d_points.append(query_keypoints[query_idx].pt)
            else:
                # XFeat numpy array
                matched_2d_points.append(query_keypoints[query_idx])
        
        return matched_map_indices, np.array(matched_2d_points, dtype=np.float32)
    
    def get_statistics(self):
        """Get matching statistics (for debugging)."""
        return {
            'ratio_threshold': self.ratio_threshold
        }