import yaml
import numpy as np
from dataclasses import dataclass
from imu.base_imu import BaseImu, ImuReading



class EurocIMU(BaseImu):
    def __init__(self, dataset_path):
        imu_yaml  = f'{dataset_path}/imu0/sensor.yaml'
        cam_yaml  = f'{dataset_path}/cam0/sensor.yaml'
        data_csv  = f'{dataset_path}/imu0/data.csv'

        data = np.loadtxt(data_csv, delimiter=',', comments='#')
        self.timestamps     = data[:, 0].astype(np.int64)
        self.gyroscope      = data[:, 1:4]
        self.accelerometer  = data[:, 4:7]

        self.T_CB       = self.load_cam_yaml(cam_yaml)
        self.imu_params = self.load_imu_yaml(imu_yaml)
        self.Q, self.R  = self.ekf_params(self.imu_params)
    
    def load_cam_yaml(self, cam_yaml):
        with open(cam_yaml, 'r') as f:
            params = yaml.safe_load(f)

            T_BS = np.array(params['T_BS']['data'], dtype=float).reshape(4, 4)

            T_CB = np.linalg.inv(T_BS)

        return T_CB
    
    def load_imu_yaml(self, imu_yaml):
        with open(imu_yaml, 'r') as f:
            params = yaml.safe_load(f)
                        
        return {
            # inertial sensor noise model parameters (static)
            'gyroscope_noise_density': float(params['gyroscope_noise_density']),     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
            'gyroscope_random_walk': float(params['gyroscope_random_walk']),    # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
            'accelerometer_noise_density': float(params['accelerometer_noise_density']),  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
            'accelerometer_random_walk': float(params['accelerometer_random_walk']),  # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )
            'time_difference': 1/params['rate_hz'] 
            }
    
    def ekf_params(self, imu_params, position_sigma = 0.01, rotation_sigma = 0.01):
        R_pos = position_sigma**2 * np.eye(3)
        R_rot = rotation_sigma**2 * np.eye(3)

        R = np.block([
            [R_pos,          np.zeros((3,3))],
            [np.zeros((3,3)), R_rot         ]
        ])

        sig_a  = imu_params['accelerometer_noise_density']
        sig_g  = imu_params['gyroscope_noise_density']
        sig_ba = imu_params['accelerometer_random_walk']
        sig_bg = imu_params['gyroscope_random_walk']
        dt = imu_params['time_difference']

        Q_pos = np.zeros((3, 3))
        Q_vel = (sig_a**2 / dt) * np.eye(3)
        Q_rot = (sig_g**2 / dt) * np.eye(3)
        Q_ba  = (sig_ba**2 * dt) * np.eye(3)
        Q_bg  = (sig_bg**2 * dt) * np.eye(3)

        Q = np.block([
            [Q_pos,           np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), Q_vel,           np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), Q_rot,           np.zeros((3,3)), np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), Q_ba,            np.zeros((3,3))],
            [np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), Q_bg          ]
        ])
        
        return Q, R 
    
    def get_readings_between(self, t_start_ns, t_end_ns):
        i_start = np.searchsorted(self.timestamps, t_start_ns, side='left')
        i_end   = np.searchsorted(self.timestamps, t_end_ns,   side='left')

        timestamps     = self.timestamps[i_start:i_end]
        gyroscope      = self.gyroscope[i_start:i_end]
        accelerometer  = self.accelerometer[i_start:i_end]

        return [
            ImuReading(
                timestamp_ns  = timestamps[i],
                gyroscope     = gyroscope[i],
                accelerometer = accelerometer[i]
            )
            for i in range(len(timestamps))
            ]
