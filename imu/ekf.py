import numpy as np
from scipy.spatial.transform import Rotation
from imu.base_imu import ImuReading


class ErrorStateEKF:

    def __init__(self, Q, R, T_CB):
        self.Q    = Q
        self.R    = R
        self.T_CB = T_CB
        self.g    = np.array([0, 0, 9.81])  # gravity cancellation vector in world frame

        # nominal state
        self.p   = np.zeros(3)
        self.v   = np.zeros(3)
        self.q   = np.array([1, 0, 0, 0], dtype=float)  # [w, x, y, z]
        self.b_a = np.zeros(3)
        self.b_g = np.zeros(3)

        self.prev_timestamp = None
        self.P = np.eye(15) * 0.1
        self.initialised = False

    def initialise(self, position, rotation_matrix, velocity=None):
        """
        Initialise state from first PnP pose.
        rotation_matrix should be the camera rotation — we convert to body frame internally.
        """
        R_CB = self.T_CB[:3, :3]
        R_body = R_CB @ rotation_matrix

        self.p = position.copy()
        self.v = velocity if velocity is not None else np.zeros(3)

        q_xyzw = Rotation.from_matrix(R_body).as_quat()  # scipy: [x,y,z,w]
        self.q  = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])  # [w,x,y,z]

        self.initialised = True

    # ── helpers ──────────────────────────────────────────────────────────────

    def _rot_vec_to_quat(self, phi):
        theta = np.linalg.norm(phi)
        if theta < 1e-8:
            return np.array([1, 0, 0, 0], dtype=float)
        w   = np.cos(theta / 2)
        xyz = (phi / theta) * np.sin(theta / 2)
        return np.concatenate([[w], xyz])

    def _quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def _quat_to_rot(self, q):
        w, x, y, z = q
        return np.array([
            [1 - 2*(y**2 + z**2),   2*(x*y - z*w),       2*(x*z + y*w)],
            [    2*(x*y + z*w),  1 - 2*(x**2 + z**2),     2*(y*z - x*w)],
            [    2*(x*z - y*w),      2*(y*z + x*w),   1 - 2*(x**2 + y**2)]
        ])

    def _skew(self, v):
        return np.array([
            [ 0,     -v[2],  v[1]],
            [ v[2],   0,    -v[0]],
            [-v[1],   v[0],  0   ]
        ])

    # ── prediction step ──────────────────────────────────────────────────────

    def predict(self, imu_reading: ImuReading):
        if not self.initialised:
            return

        gyroscope_corrected     = imu_reading.gyroscope     - self.b_g
        accelerometer_corrected = imu_reading.accelerometer - self.b_a

        if self.prev_timestamp is None:
            self.prev_timestamp = imu_reading.timestamp_ns
            return

        current_timestamp = imu_reading.timestamp_ns
        dt = (current_timestamp - self.prev_timestamp) / 1e9

        R = self._quat_to_rot(self.q)

        # rotate accelerometer to world frame and subtract gravity
        a_world = R @ accelerometer_corrected + self.g

        # propagate nominal state
        self.p = self.p + self.v * dt + 0.5 * a_world * dt**2
        self.v = self.v + a_world * dt
        self.q = self._quat_multiply(self.q, self._rot_vec_to_quat(gyroscope_corrected * dt))
        self.q = self.q / np.linalg.norm(self.q)

        self.prev_timestamp = current_timestamp

        # error state Jacobian
        I = np.eye(3)
        Z = np.zeros((3, 3))

        F = np.block([
            [I,    I*dt,  Z,                                          Z,       Z     ],
            [Z,    I,    -R @ self._skew(accelerometer_corrected)*dt, -R*dt,   Z     ],
            [Z,    Z,     I,                                          Z,      -I*dt  ],
            [Z,    Z,     Z,                                          I,       Z     ],
            [Z,    Z,     Z,                                          Z,       I     ]
        ])

        self.P = F @ self.P @ F.T + self.Q

    # ── update step ──────────────────────────────────────────────────────────

    def update(self, position, rotation_matrix):
        """
        Update EKF with a visual pose.
        position:        camera centre in world frame (metres)
        rotation_matrix: camera rotation matrix from PnP (R_cam)
        """
        if not self.initialised:
            return

        # convert visual pose from camera frame to IMU body frame
        R_CB    = self.T_CB[:3, :3]
        t_CB    = self.T_CB[:3, 3]
        position_body    = R_CB @ position + t_CB
        rotation_body    = R_CB @ rotation_matrix

        # innovation
        delta_p     = position_body - self.p
        R_nominal   = self._quat_to_rot(self.q)
        delta_R     = R_nominal.T @ rotation_body
        delta_theta = Rotation.from_matrix(delta_R).as_rotvec()
        z = np.concatenate([delta_p, delta_theta])  # (6,)

        I    = np.eye(3)
        zero = np.zeros((3, 3))

        H = np.block([
            [I,    zero, zero, zero, zero],
            [zero, zero, I,    zero, zero]
        ])  # (6, 15)

        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = (np.linalg.solve(S.T, H @ self.P.T)).T  # (15, 6)

        # error state estimate
        delta_x = K @ z

        # inject error into nominal state
        self.p   = self.p   + delta_x[0:3]
        self.v   = self.v   + delta_x[3:6]
        self.b_a = self.b_a + delta_x[9:12]
        self.b_g = self.b_g + delta_x[12:15]

        orient_error_quat = self._rot_vec_to_quat(delta_x[6:9])
        self.q = self._quat_multiply(self.q, orient_error_quat)
        self.q = self.q / np.linalg.norm(self.q)

        # Joseph form covariance update
        I_KH   = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T

    # ── output ───────────────────────────────────────────────────────────────

    def get_state(self):
        return self.p, self._quat_to_rot(self.q)