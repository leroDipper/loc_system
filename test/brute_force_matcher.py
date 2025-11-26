"""
brute_force_matcher.py

Brute-force matchers for benchmarking vocabulary tree speedup.
Includes both pure Python and OpenCV (C++) implementations.
"""

import numpy as np
import cv2

class BruteForceMatcher:
    """
    Exhaustive nearest-neighbor matcher with Lowe's ratio test (Pure Python).
    O(N×M) complexity - compares every query against every map descriptor.
    """
    
    def __init__(self, map_descriptors):
        """
        Initialize with map descriptors.
        
        Args:
            map_descriptors: numpy array (M, D) where M = number of map descriptors
        """
        self.map_descriptors = map_descriptors.astype(np.float32)
        self.M = len(map_descriptors)
        print(f"BruteForceMatcher (Python) initialized with {self.M} map descriptors")
    
    def match(self, query_descriptors, ratio_threshold=0.80):
        """
        Match query descriptors against map descriptors using brute-force search.
        
        Args:
            query_descriptors: numpy array (N, D)
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            tuple: (query_indices, map_indices, distances)
        """
        query_descriptors = query_descriptors.astype(np.float32)
        N = len(query_descriptors)
        
        match_query_idx = []
        match_map_idx = []
        match_distances = []
        
        # For each query descriptor
        for q_idx in range(N):
            query_desc = query_descriptors[q_idx]
            
            # Compute distances to ALL map descriptors (brute force)
            distances = np.linalg.norm(
                self.map_descriptors - query_desc, 
                axis=1
            )
            
            # Find two nearest neighbors efficiently
            if len(distances) < 2:
                continue
            
            # Use argpartition to find top 2 (much faster than full sort)
            top2_indices = np.argpartition(distances, 1)[:2]
            top2_distances = distances[top2_indices]
            
            # Sort just these 2
            sorted_order = np.argsort(top2_distances)
            best_idx = top2_indices[sorted_order[0]]
            second_best_idx = top2_indices[sorted_order[1]]
            
            best_dist = distances[best_idx]
            second_best_dist = distances[second_best_idx]
            
            # Lowe's ratio test
            if best_dist < ratio_threshold * second_best_dist:
                match_query_idx.append(q_idx)
                match_map_idx.append(best_idx)
                match_distances.append(best_dist)
        
        return (
            np.array(match_query_idx, dtype=np.int32),
            np.array(match_map_idx, dtype=np.int32),
            np.array(match_distances, dtype=np.float32)
        )


class OpenCVBruteForceMatcher:
    """
    OpenCV's BFMatcher (C++ implementation) for fair comparison.
    Also O(N×M) but optimized C++ code.
    """
    
    def __init__(self, map_descriptors):
        """
        Initialize with map descriptors.
        
        Args:
            map_descriptors: numpy array (M, D)
        """
        self.map_descriptors = map_descriptors.astype(np.float32)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.M = len(map_descriptors)
        print(f"OpenCV BFMatcher initialized with {self.M} map descriptors")
    
    def match(self, query_descriptors, ratio_threshold=0.80):
        """
        Match using OpenCV BFMatcher with ratio test.
        
        Args:
            query_descriptors: numpy array (N, D)
            ratio_threshold: Lowe's ratio test threshold
            
        Returns:
            tuple: (query_indices, map_indices, distances)
        """
        query_descriptors = query_descriptors.astype(np.float32)
        
        # knnMatch returns top 2 matches for each query
        matches = self.matcher.knnMatch(query_descriptors, self.map_descriptors, k=2)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        # Extract indices and distances
        query_idx = np.array([m.queryIdx for m in good_matches], dtype=np.int32)
        map_idx = np.array([m.trainIdx for m in good_matches], dtype=np.int32)
        distances = np.array([m.distance for m in good_matches], dtype=np.float32)
        
        return query_idx, map_idx, distances