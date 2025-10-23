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
            map_descriptors: N×128 array of map SIFT descriptors
            query_descriptors: M×128 array of query SIFT descriptors
            query_keypoints: List of query cv2.KeyPoint objects
            
        Returns:
            tuple: (matched_3d_indices, matched_2d_points) where:
                - matched_3d_indices: List of map point indices
                - matched_2d_points: N×2 array of corresponding 2D points
        """
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
            matched_2d_points.append(query_keypoints[query_idx].pt)
        
        return matched_map_indices, np.array(matched_2d_points, dtype=np.float32)
    
    def get_statistics(self):
        """Get matching statistics (for debugging)."""
        return {
            'ratio_threshold': self.ratio_threshold
        }