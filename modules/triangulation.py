import cv2
import numpy as np

class Triangulator:
    def __init__(self, camera_matrix):
        self.K = camera_matrix
        
    def triangulate_points(self, pts1, pts2, R1, t1, R2, t2):
        # Projection matrices
        P1 = self.K @ np.hstack([R1, t1.reshape(-1, 1)])
        P2 = self.K @ np.hstack([R2, t2.reshape(-1, 1)])
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]  # Convert from homogeneous
        
        return points_3d.T
    
    def get_scale(self, points_3d_prev, points_3d_curr, R, t):
        # Simple scale estimation from triangulated points
        if points_3d_prev is None or len(points_3d_prev) < 10:
            return 1.0
        
        if len(points_3d_curr) < 10:
            return 1.0
            
        # Method 1: Compare median depths (simple but effective)
        depth_prev = np.median(points_3d_prev[:, 2])
        depth_curr = np.median(points_3d_curr[:, 2])
        
        if depth_curr > 0 and depth_prev > 0:
            scale_from_depth = depth_prev / depth_curr
        else:
            scale_from_depth = 1.0
        
        # Method 2: Compare scene scales (distances from origin)
        dist_prev = np.median(np.linalg.norm(points_3d_prev, axis=1))
        dist_curr = np.median(np.linalg.norm(points_3d_curr, axis=1))
        
        if dist_curr > 0 and dist_prev > 0:
            scale_from_distance = dist_prev / dist_curr
        else:
            scale_from_distance = 1.0
        
        # Combine the two methods
        scale = 0.7 * scale_from_depth + 0.3 * scale_from_distance
        
        # Apply reasonable bounds
        scale = np.clip(scale, 0.1, 10.0)
        
        return scale

# Legacy functions if you need them elsewhere
def triangulate(P1, P2, pts1, pts2):
    """Simple triangulation function"""
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T

def triangulate_unfiltered(P1, P2, pts1, pts2):
    """Unfiltered triangulation function"""
    return triangulate(P1, P2, pts1, pts2)