import cv2
import numpy as np

class SIFT:
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.sift = cv2.SIFT_create(nfeatures=self.n_features)

    def convert_to_grayscale(self, image):
        """Convert an image to grayscale if it is not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    

    def detect_and_compute(self, image, mask=None):
        """Detect keypoints and compute descriptors."""
        gray_image = self.convert_to_grayscale(image)
        keypints, descriptors = self.sift.detectAndCompute(gray_image, mask)
        return keypints, descriptors
    
    def match_features(self, left_descriptors, right_descriptors):
        """Match features between two sets of descriptors using Lowe's ratio test."""
        if left_descriptors is None or right_descriptors is None:
            return []

        # FLANN or BFMatcher - here we use BFMatcher for simplicity
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(left_descriptors, right_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


        return good_matches
    

class BRISK:
    def __init__(self, thresh=20, octaves=3, patternScale=1.2):
        self.thresh = thresh
        self.octaves = octaves
        self.patternScale = patternScale
        self.brisk = cv2.BRISK_create(thresh=self.thresh, octaves=self.octaves, patternScale=self.patternScale)

    def convert_to_grayscale(self, image):
        """Convert an image to grayscale if it is not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def detect_and_compute(self, image, mask=None):
        """Detect keypoints and compute descriptors."""
        gray_image = self.convert_to_grayscale(image)
        keypoints, descriptors = self.brisk.detectAndCompute(gray_image, mask)
        return keypoints, descriptors

    def match_features(self, left_descriptors, right_descriptors):
        """Match features using Hamming distance + Lowe's ratio test."""
        if left_descriptors is None or right_descriptors is None:
            return []

        # Use Hamming distance for binary descriptors like BRISK
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(left_descriptors, right_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        return good_matches




class ORB:
    def __init__(self, n_features=500):
        self.n_features = n_features
        self.orb = cv2.ORB_create(nfeatures=self.n_features)

    def convert_to_grayscale(self, image):
        """Convert an image to grayscale if it is not already."""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def detect_and_compute(self, image, mask=None):
        """Detect keypoints and compute descriptors."""
        gray_image = self.convert_to_grayscale(image)
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, mask)
        return keypoints, descriptors

    def match_features(self, left_descriptors, right_descriptors):
        """Match features using Hamming distance + Lowe's ratio test."""
        if left_descriptors is None or right_descriptors is None:
            return []

        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(left_descriptors, right_descriptors, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)


        return good_matches

        
