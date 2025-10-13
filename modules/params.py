import cv2
import numpy as np
import yaml



def load_calibration(filepath):
    with open(filepath, 'r') as file:
        data = yaml.safe_load(file)
        fx, fy, cx, cy = data['intrinsics']
        K = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0,  0,  1]])
        D = np.array(data['distortion_coefficients'])
        T_BS = np.array(data['T_BS']['data']).reshape(4, 4)

        return K, D, T_BS,

