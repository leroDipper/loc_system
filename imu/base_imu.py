import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class ImuReading:
    timestamp_ns: int
    gyroscope: np.ndarray      # (3,) rad/s  — [wx, wy, wz]
    accelerometer: np.ndarray  # (3,) m/s²   — [ax, ay, az]

class BaseImu(ABC):
    
    @abstractmethod
    def get_readings_between(self, t_start_ns: int, t_end_ns: int) -> list[ImuReading]:
        """Return all IMU readings where t_start_ns <= t < t_end_ns."""
        pass