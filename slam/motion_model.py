import numpy as np

class MotionModel:
    def __init__(self):
        self.velocity = None

    def update(self, T_current, T_last):
        self.velocity = T_current @ np.linalg.inv(T_last)
       
    
    def predict(self, T_last):
        if self.velocity is None:
            return None
        return self.velocity @ T_last
    
    def invalidate(self):
        self.velocity = None


