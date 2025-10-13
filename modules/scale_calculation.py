import numpy as np

def calculate_scale_from_markers(marker_poses, vo_translation):
    """Calculate scale from first 2 marker detections"""
    if len(marker_poses) < 2:
        print(f"Not enough marker poses: {len(marker_poses)}")
        return None
    
    #print(f"Marker pose 0: {marker_poses[0]}")
    #print(f"Marker pose 1: {marker_poses[1]}")

    #print(f"VO poses: {vo_translation}")
    
    
    real_baseline = np.linalg.norm(marker_poses[1] - marker_poses[0])
    vo_baseline = np.linalg.norm(vo_translation)
    
    #print(f"Real baseline: {real_baseline:.6f}m")
    #print(f"VO baseline: {vo_baseline:.6f}")
    
    if vo_baseline < 1e-6:
        return None
        
    scale = real_baseline / vo_baseline
    return scale