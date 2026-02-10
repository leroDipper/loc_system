"""
Pose Uncertainty Estimation Module - Bayesian Version

Uses hierarchical Bayesian model to estimate pose uncertainty based on 
frame-level features rather than per-match features.
"""

import numpy as np
import joblib
from typing import Tuple, Dict, Optional
import warnings
import cv2


class BayesianPoseUncertaintyEstimator:
    """Estimates pose uncertainty using hierarchical Bayesian model."""
    
    def __init__(
        self,
        model_path: str = "results/bayesian_model.joblib",
        use_dataset_specific: bool = False,
        dataset_name: Optional[str] = None
    ):
        """
        Initialize the Bayesian uncertainty estimator.
        
        Args:
            model_path: Path to trained Bayesian model
            use_dataset_specific: If True, use dataset-specific parameters
            dataset_name: Which dataset to use if dataset_specific is True
                         (one of: 'FR1', 'FR3', 'MH_01', 'MH_03')
        """
        self.use_dataset_specific = use_dataset_specific
        self.dataset_name = dataset_name
        
        # Load trained Bayesian model
        try:
            model_data = joblib.load(model_path)
            
            # Extract model components
            self.feature_names = model_data['features']
            self.scaler = model_data['scaler']
            self.y_mean = model_data['y_mean']
            self.y_std = model_data['y_std']
            
            # Global parameters (always available)
            self.μ_global = model_data['μ_global']
            self.sigma_global = model_data['sigma_global']
            
            # Dataset-specific parameters (optional)
            if use_dataset_specific:
                self.β = model_data['β']  # (n_datasets, n_features)
                self.dataset_names = model_data['dataset_names']
                
                if dataset_name is not None:
                    if dataset_name not in self.dataset_names:
                        raise ValueError(
                            f"Dataset '{dataset_name}' not found. "
                            f"Available: {self.dataset_names}"
                        )
                    self.dataset_idx = self.dataset_names.index(dataset_name)
                else:
                    warnings.warn(
                        "Dataset-specific mode enabled but no dataset specified. "
                        "Will use global priors."
                    )
                    self.use_dataset_specific = False
            
            print(f" Loaded Bayesian uncertainty model from {model_path}")
            print(f"  R²: {model_data.get('r2', 'N/A'):.3f}")
            print(f"  RMSE: {model_data.get('rmse', 'N/A')*100:.2f} cm")
            print(f"  Features: {len(self.feature_names)}")
            if use_dataset_specific and dataset_name:
                print(f"  Using {dataset_name}-specific parameters")
            else:
                print(f"  Using global parameters")
                
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Bayesian model not found at {model_path}. "
                f"Train it first using train_bayesian.py"
            )
    
    def extract_frame_features(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        inlier_points_3d: np.ndarray,
        inlier_points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        reprojection_errors: np.ndarray,
        image_gray: Optional[np.ndarray] = None,
        all_keypoints: Optional[np.ndarray] = None,
        n_raw_matches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Extract frame-level features for Bayesian model.
        
        This matches the features used in train_bayesian.py:
        - mean_inverse_depth
        - mean_n_candidates
        - mean_inlier_ba_error (reprojection error)
        - depth_mean
        - match_spread_normalized
        - match_std_x
        - quadrant_entropy
        - n_inliers
        - depth_relative_std
        
        Args:
            points_3d: All matched 3D points before RANSAC (N x 3)
            points_2d: All matched 2D points before RANSAC (N x 2)
            inlier_points_3d: Inlier 3D points after RANSAC (M x 3)
            inlier_points_2d: Inlier 2D points after RANSAC (M x 2)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3 x 1)
            reprojection_errors: Per-inlier reprojection errors (M,)
            image_gray: Optional grayscale image for image quality features
            all_keypoints: Optional all detected keypoints for spatial analysis
            n_raw_matches: Optional total number of raw matches before filtering
            
        Returns:
            features: Dictionary with feature values
        """
        features = {}
        
        # Transform 3D points to camera frame
        points_camera = (R @ inlier_points_3d.T).T + t.reshape(1, 3)
        depths = points_camera[:, 2]  # Z-coordinate is depth
        
        # 1. mean_inverse_depth
        inverse_depths = 1.0 / (depths + 1e-6)
        features['mean_inverse_depth'] = np.mean(inverse_depths)
        
        # 2. depth_mean
        features['depth_mean'] = np.mean(depths)
        
        # 3. depth_relative_std
        features['depth_relative_std'] = np.std(depths) / (np.mean(depths) + 1e-6)
        
        # 4. mean_inlier_ba_error (reprojection error)
        features['mean_inlier_ba_error'] = np.mean(reprojection_errors)
        
        # 5. n_inliers
        features['n_inliers'] = len(inlier_points_3d)
        
        # 6. match_std_x (spatial spread of matches in X)
        if camera_matrix is not None:
            img_width = 2 * camera_matrix[0, 2]
            features['match_std_x'] = np.std(inlier_points_2d[:, 0]) / img_width
        else:
            raise ValueError("camera_matrix is required for feature extraction")
        #     features['match_std_x'] = np.std(inlier_points_2d[:, 0]) / 752.0
        
        # 7. match_spread_normalized (spatial coverage)
        if len(inlier_points_2d) >= 3:
            # Use minimum enclosing circle (same as training)
            center, radius = cv2.minEnclosingCircle(inlier_points_2d.astype(np.float32))
            
            if camera_matrix is not None:
                img_width = 2 * camera_matrix[0, 2]
                img_height = 2 * camera_matrix[1, 2]
                diagonal = np.sqrt(img_width**2 + img_height**2)
                features['match_spread_normalized'] = radius / diagonal
            else:
                # Fallback
                diagonal = np.sqrt(752**2 + 480**2)
                features['match_spread_normalized'] = radius / diagonal
        else:
            features['match_spread_normalized'] = 0.0
        # 8. quadrant_entropy (distribution across image quadrants)
        if camera_matrix is not None:
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]
            
            # Count points in each quadrant
            quadrant_counts = np.zeros(4)
            for point in inlier_points_2d:
                x, y = point
                if x < cx and y < cy:
                    quadrant_counts[0] += 1  # Top-left
                elif x >= cx and y < cy:
                    quadrant_counts[1] += 1  # Top-right
                elif x < cx and y >= cy:
                    quadrant_counts[2] += 1  # Bottom-left
                else:
                    quadrant_counts[3] += 1  # Bottom-right
            
            # Compute entropy
            total = np.sum(quadrant_counts)
            if total > 0:
                probs = quadrant_counts / total
                probs = probs[probs > 0]  # Remove zeros
                features['quadrant_entropy'] = -np.sum(probs * np.log2(probs + 1e-10))
            else:
                features['quadrant_entropy'] = 0.0
        else:
            features['quadrant_entropy'] = 0.0
        
        return features
    
    def predict_error(
        self,
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Predict localization error using Bayesian model.
        
        Args:
            features: Dictionary of frame-level features
            
        Returns:
            mean_error: Predicted mean error (meters)
            std_error: Predicted standard deviation (meters)
        """
        # Build feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        for i, feat_name in enumerate(self.feature_names):
            if feat_name in features:
                feature_vector[i] = features[feat_name]
            else:
                warnings.warn(f"Feature '{feat_name}' not available, using 0")
                feature_vector[i] = 0.0
        
        # Standardize features using same scaler as training
        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))[0]
        
        # Clip extreme values (same as training)
        feature_vector_scaled = np.clip(feature_vector_scaled, -5, 5)
        
        # Predict using appropriate parameters
        if self.use_dataset_specific and hasattr(self, 'dataset_idx'):
            # Use dataset-specific effects
            β = self.β[self.dataset_idx]
            error_scaled = np.dot(β, feature_vector_scaled)
        else:
            # Use global priors
            error_scaled = np.dot(self.μ_global, feature_vector_scaled)
        
        # Unscale to meters
        mean_error = error_scaled * self.y_std + self.y_mean
        
        # Uncertainty from global variance
        # This represents epistemic uncertainty (model uncertainty)
        std_error = np.sqrt(np.sum((self.sigma_global * feature_vector_scaled)**2))
        std_error = std_error * self.y_std  # Unscale
        
        # Ensure positive error
        mean_error = max(0.001, mean_error)  # At least 1mm
        std_error = max(0.001, std_error)
        
        return mean_error, std_error
    
    def error_to_pixel_uncertainty(
        self,
        predicted_error: float,
        camera_matrix: np.ndarray,
        average_depth: float
    ) -> float:
        """
        Convert predicted spatial error (meters) to pixel uncertainty.
        
        Uses simple pinhole camera model:
        pixel_error ≈ focal_length * spatial_error / depth
        
        Args:
            predicted_error: Predicted error in meters
            camera_matrix: Camera intrinsic matrix
            average_depth: Average depth of matched points (meters)
            
        Returns:
            pixel_uncertainty: Uncertainty in pixels
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        f_avg = (fx + fy) / 2.0
        
        # Convert spatial error to pixel error
        pixel_uncertainty = f_avg * predicted_error / (average_depth + 1e-6)
        
        # Clamp to reasonable range
        pixel_uncertainty = np.clip(pixel_uncertainty, 0.5, 50.0)
        
        return pixel_uncertainty
    
    def compute_pose_covariance(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        pixel_uncertainty: float
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute 6-DOF pose covariance from predicted uncertainty.
        
        Uses first-order uncertainty propagation with uniform pixel uncertainty
        for all matches (since Bayesian model gives frame-level prediction).
        
        Args:
            points_3d: 3D points in world coordinates (N x 3)
            points_2d: 2D observations in image (N x 2)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3 x 1)
            pixel_uncertainty: Predicted pixel uncertainty (same for all points)
            
        Returns:
            covariance: 6x6 pose covariance matrix (rotation + translation)
            info: Dictionary with diagnostic information
        """
        n_points = len(points_3d)
        
        # Transform points to camera frame
        points_camera = (R @ points_3d.T).T + t.reshape(1, 3)
        
        # Extract camera parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        # Build Jacobian matrix (2N x 6)
        J = np.zeros((2 * n_points, 6))
        
        for i in range(n_points):
            X, Y, Z = points_camera[i]
            Z2 = Z * Z
            
            # Jacobian of projection w.r.t. camera-frame point
            du_dX = fx / Z
            du_dY = 0
            du_dZ = -fx * X / Z2
            
            dv_dX = 0
            dv_dY = fy / Z
            dv_dZ = -fy * Y / Z2
            
            # Jacobian w.r.t. rotation (small angle approximation)
            J[2*i, 0] = du_dX * 0      + du_dY * Z      + du_dZ * (-Y)
            J[2*i, 1] = du_dX * (-Z)   + du_dY * 0      + du_dZ * X
            J[2*i, 2] = du_dX * Y      + du_dY * (-X)   + du_dZ * 0
            
            J[2*i+1, 0] = dv_dX * 0    + dv_dY * Z      + dv_dZ * (-Y)
            J[2*i+1, 1] = dv_dX * (-Z) + dv_dY * 0      + dv_dZ * X
            J[2*i+1, 2] = dv_dX * Y    + dv_dY * (-X)   + dv_dZ * 0
            
            # Jacobian w.r.t. translation
            J[2*i, 3]   = du_dX
            J[2*i, 4]   = du_dY
            J[2*i, 5]   = du_dZ
            
            J[2*i+1, 3] = dv_dX
            J[2*i+1, 4] = dv_dY
            J[2*i+1, 5] = dv_dZ
        
        # Build weight matrix (uniform weights from Bayesian prediction)
        weight = 1.0 / (pixel_uncertainty ** 2)
        W = np.eye(2 * n_points) * weight
        
        # Compute information matrix: H = J^T W J
        H = J.T @ W @ J
        
        # Compute covariance: Σ = H^{-1}
        try:
            covariance = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            warnings.warn("Information matrix is singular, using pseudo-inverse")
            covariance = np.linalg.pinv(H)
        
        # Compute diagnostics
        points_proj_homo = camera_matrix @ points_camera.T
        points_proj = (points_proj_homo[:2, :] / points_proj_homo[2, :]).T
        reprojection_errors = np.linalg.norm(points_2d - points_proj, axis=1)
        
        info = {
            "n_points": n_points,
            "pixel_uncertainty": pixel_uncertainty,
            "mean_reprojection_error": np.mean(reprojection_errors),
            "median_reprojection_error": np.median(reprojection_errors),
            "condition_number": np.linalg.cond(H),
            "rotation_uncertainty": np.sqrt(np.diag(covariance[:3, :3])),
            "translation_uncertainty": np.sqrt(np.diag(covariance[3:, 3:]))
        }
        
        return covariance, info
    
    def estimate_pose_uncertainty(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        inlier_points_3d: np.ndarray,
        inlier_points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        reprojection_errors: np.ndarray,
        image_gray: Optional[np.ndarray] = None,
        all_keypoints: Optional[np.ndarray] = None,
        n_raw_matches: Optional[int] = None,
        skip_covariance: bool = False
    ) -> Dict:
        """
        End-to-end pose uncertainty estimation using Bayesian model.
        
        Args:
            points_3d: All matched 3D points before RANSAC (N x 3)
            points_2d: All matched 2D points before RANSAC (N x 2)
            inlier_points_3d: Inlier 3D points after RANSAC (M x 3)
            inlier_points_2d: Inlier 2D points after RANSAC (M x 2)
            camera_matrix: Camera intrinsics (3 x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3 x 1)
            reprojection_errors: Per-inlier reprojection errors (M,)
            image_gray: Optional grayscale image
            all_keypoints: Optional all detected keypoints
            n_raw_matches: Optional total raw matches
            
        Returns:
            result: Dictionary containing:
                - covariance: 6x6 pose covariance
                - predicted_error_m: Predicted mean error (meters)
                - predicted_error_std: Predicted error std dev (meters)
                - pixel_uncertainty: Predicted pixel uncertainty
                - features: Extracted features
                - info: Diagnostic information
        """
        # Step 1: Extract frame-level features
        features = self.extract_frame_features(
            points_3d,
            points_2d,
            inlier_points_3d,
            inlier_points_2d,
            camera_matrix,
            R,
            t,
            reprojection_errors,
            image_gray,
            all_keypoints,
            n_raw_matches
        )
        
        # Step 2: Predict error using Bayesian model
        predicted_error_m, predicted_error_std = self.predict_error(features)
        
        # Step 3: Convert to pixel uncertainty
        average_depth = features['depth_mean']
        pixel_uncertainty = self.error_to_pixel_uncertainty(
            predicted_error_m,
            camera_matrix,
            average_depth
        )
        
        # Step 4: Compute pose covariance (optional - can skip for speed)
        if not skip_covariance:
            covariance, info = self.compute_pose_covariance(
                inlier_points_3d,
                inlier_points_2d,
                camera_matrix,
                R,
                t,
                pixel_uncertainty
            )
            # Add Bayesian predictions to info
            info['predicted_error_m'] = predicted_error_m
            info['predicted_error_std'] = predicted_error_std
            info['features'] = features
        else:
            # Skip covariance computation for speed
            covariance = None
            info = {
                'predicted_error_m': predicted_error_m,
                'predicted_error_std': predicted_error_std,
                'features': features,
                'pixel_uncertainty': pixel_uncertainty,
                'mean_reprojection_error': np.mean(reprojection_errors)
            }
        
        return {
            "covariance": covariance,
            "predicted_error_m": predicted_error_m,
            "predicted_error_std": predicted_error_std,
            "predicted_error_cm": predicted_error_m * 100,
            "predicted_std_cm": predicted_error_std * 100,
            "pixel_uncertainty": pixel_uncertainty,
            "features": features,
            "info": info
        }


def print_bayesian_uncertainty_summary(result: Dict, actual_error: Optional[float] = None):
    """
    Print a human-readable summary of Bayesian uncertainty estimation.
    
    Args:
        result: Output from estimate_pose_uncertainty()
        actual_error: Optional ground truth error for validation
    """
    print("\n" + "="*60)
    print("BAYESIAN POSE UNCERTAINTY ESTIMATE")
    print("="*60)
    
    info = result['info']
    
    print(f"\nPredicted Error (Bayesian Model):")
    print(f"  Mean:  {result['predicted_error_cm']:.2f} cm")
    print(f"  Std:   {result['predicted_error_std']*100:.2f} cm")
    print(f"  Range: [{(result['predicted_error_m'] - result['predicted_error_std'])*100:.2f}, "
          f"{(result['predicted_error_m'] + result['predicted_error_std'])*100:.2f}] cm")
    
    print(f"\nFrame Features:")
    print(f"  Number of inliers:      {int(info['features']['n_inliers'])}")
    print(f"  Mean depth:             {info['features']['depth_mean']:.2f} m")
    print(f"  Mean reproj error:      {info['features']['mean_inlier_ba_error']:.2f} px")
    print(f"  Match spread:           {info['features']['match_spread_normalized']:.3f}")
    print(f"  Quadrant entropy:       {info['features']['quadrant_entropy']:.2f}")
    
    print(f"\nPose Uncertainty:")
    rot_std = info['rotation_uncertainty']
    trans_std = info['translation_uncertainty']
    print(f"  Rotation std:     [{rot_std[0]:.4f}, {rot_std[1]:.4f}, {rot_std[2]:.4f}] rad")
    print(f"  Translation std:  [{trans_std[0]:.4f}, {trans_std[1]:.4f}, {trans_std[2]:.4f}] m")
    print(f"  Total trans. unc: {np.linalg.norm(trans_std)*100:.2f} cm")
    
    print(f"\nDiagnostics:")
    print(f"  Pixel uncertainty:       {info['pixel_uncertainty']:.2f} px")
    print(f"  Mean reprojection error: {info['mean_reprojection_error']:.2f} px")
    print(f"  Information matrix cond: {info['condition_number']:.1e}")
    
    if actual_error is not None:
        print(f"\nValidation:")
        print(f"  Predicted error:     {result['predicted_error_cm']:.2f} cm")
        print(f"  Actual error:        {actual_error*100:.2f} cm")
        print(f"  Residual:            {(actual_error - result['predicted_error_m'])*100:.2f} cm")
        print(f"  Within 1sigma?           {abs(actual_error - result['predicted_error_m']) <= result['predicted_error_std']}")
        
        # Check if actual error is within predicted range
        lower = result['predicted_error_m'] - result['predicted_error_std']
        upper = result['predicted_error_m'] + result['predicted_error_std']
        within_range = lower <= actual_error <= upper
        print(f"  Within predicted range? {within_range}")
    
    print("="*60)