import numpy as np

class KeyFrameDecision:
    def __init__(self):
        self.min_frames     = 20    
        self.ref_ratio      = 0.9   
        self.min_static     = 20   
        self.min_total      = 15  

        self.last_kf_frame_id  = 0   
        self.ref_inliers       = 0   

    def decide(self, frame_id, static_inliers, total_inliers):
        condition_1 = frame_id >= self.last_kf_frame_id + self.min_frames
        condition_2 = static_inliers < self.ref_inliers*self.ref_ratio
        condition_3 = static_inliers < self.min_static

        need_keyframe = ((condition_1 and condition_2) or condition_3) and total_inliers >= self.min_total
        if need_keyframe:
            self.last_kf_frame_id = frame_id
            self.ref_inliers = total_inliers

        return need_keyframe

