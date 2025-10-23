"""Feature extraction functionality."""

import cv2
import numpy as np


class FeatureExtractor:
    """Extracts SIFT features from images."""
    
    def __init__(self):
        """Initialize SIFT extractor."""
        self.sift = cv2.SIFT_create()
    
    def extract(self, image_path):
        """
        Extract SIFT features from an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            tuple: (keypoints, descriptors) where:
                - keypoints: List of cv2.KeyPoint objects
                - descriptors: N×128 array of SIFT descriptors
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect and compute
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        return keypoints, descriptors
    
    def extract_from_array(self, img_array):
        """
        Extract SIFT features from an image array.
        
        Args:
            img_array: Image as numpy array (BGR format)
            
        Returns:
            tuple: (keypoints, descriptors)
        """
        gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors