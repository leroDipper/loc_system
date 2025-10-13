import cv2
import numpy as np
from modules.frame_loader import FrameLoader

class Rectification:
    def __init__(self, images_path, max_images=None):
        self.frame_loader = FrameLoader(images_path, max_images)

    def rectify(self, K_left, K_right, D_left, D_right, R, T, image_size):
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K_left, D_left, K_right, D_right, image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
        )

        # Extract rectified camera matrices from projection matrices
        # P1 = [K_left_rectified | 0] for left camera
        # P2 = [K_right_rectified | K_right_rectified * baseline] for right camera
        self.K_left_rectified = P1[:3, :3]
        self.K_right_rectified = P2[:3, :3]
        
        # After rectification, distortion should be zero
        self.D_left_rectified = np.zeros(5)
        self.D_right_rectified = np.zeros(5)


        left_map1, left_map2 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)
        return left_map1, left_map2, right_map1, right_map2, P1, P2

    def get_rectified_frames(self, K_left, K_right, D_left, D_right, R, T, image_size):
        left_map1, left_map2, right_map1, right_map2, P1, P2 = self.rectify(
            K_left, K_right, D_left, D_right, R, T, image_size
        )

        for left_image, right_image, timestamp in self.frame_loader.get_frame_pairs_with_timestamps():
            rectified_left = cv2.remap(left_image, left_map1, left_map2, cv2.INTER_LINEAR)
            rectified_right = cv2.remap(right_image, right_map1, right_map2, cv2.INTER_LINEAR)
            yield rectified_left, rectified_right, timestamp, P1, P2
        # Note: P1 and P2 are returned for triangulation purposes


    def get_rectified_camera_params(self):
        """
        Get the rectified camera parameters for pose estimation
        
        Returns:
            K_left_rect, D_left_rect, K_right_rect, D_right_rect
        """
        if self.K_left_rectified is None:
            raise ValueError("Must call rectify() first to compute rectified parameters")
            
        return (self.K_left_rectified, self.D_left_rectified, 
                self.K_right_rectified, self.D_right_rectified)
