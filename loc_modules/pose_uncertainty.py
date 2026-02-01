"""
Pose Uncertainty Estimation Module - Version 2 (with Geometric Features)

Converts match-level confidence scores into pose covariance using:
1. ML model: match features + geometric features → confidence p
2. Uncertainty scaling: p → pixel uncertainty σ
3. Covariance propagation: σ → pose covariance Σ

NEW: Now includes geometric quality features for improved uncertainty prediction.
"""

import numpy as np
import joblib
from typing import Tuple, Dict, Optional
import warnings


class PoseUncertaintyEstimator:
    """Estimates pose covariance from match confidence and geometry."""
    
    def __init__(
        self,
        model_path: str = "results/match_confidence_model.joblib",
        sigma_base: float = 1.0,
        min_confidence: float = 0.3,
        use_reprojection_refinement: bool = True
    ):
        """
        Initialize the uncertainty estimator.
        
        Args:
            model_path: Path to trained match confidence model
            sigma_base: Base pixel uncertainty for perfect matches (pixels)
            min_confidence: Minimum confidence threshold (reject below this)
            use_reprojection_refinement: Whether to refine uncertainty with reprojection error
        """
        self.sigma_base = sigma_base
        self.min_confidence = min_confidence
        self.use_reprojection_refinement = use_reprojection_refinement
        
        # Load trained model
        try:
            model_data = joblib.load(model_path)
            self.model = model_data["model"]
            self.feature_names = model_data["features"]
            print(f"✓ Loaded match confidence model from {model_path}")
            print(f"  AUC: {model_data.get('auc_test', 'N/A'):.3f}")
            print(f"  Features: {', '.join(self.feature_names)}")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Train it first using train_descent_geometric.py"
            )
    
    def predict_match_confidence(
        self,
        best_match_distances: np.ndarray,
        ratio_test_values: np.ndarray,
        n_candidates: np.ndarray,
        reprojection_errors: np.ndarray,
        detector_scores: np.ndarray,
        geometric_features: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Predict confidence for each match using the ML model.
        
        Args:
            best_match_distances: Descriptor distance to best match
            ratio_test_values: Lowe's ratio test values
            n_candidates: Number of candidate matches per feature
            reprojection_errors: Geometric reprojection errors (pixels)
            detector_scores: XFeat detector confidence scores
            geometric_features: Dict with global geometric quality features (NEW)
            
        Returns:
            confidence: Array of confidence scores [0, 1] for each match
        """
        # Build feature matrix
        n_matches = len(best_match_distances)
        X = np.zeros((n_matches, len(self.feature_names)))
        
        for i, feat_name in enumerate(self.feature_names):
            # Per-match features (used directly)
            if feat_name == "best_match_distance":
                X[:, i] = best_match_distances
            elif feat_name == "ratio_test_value":
                X[:, i] = ratio_test_values
            elif feat_name == "n_candidates":
                X[:, i] = n_candidates
            elif feat_name == "reprojection_error":
                X[:, i] = reprojection_errors
            elif feat_name == "detector_score":
                X[:, i] = detector_scores
            
            # Aggregate features computed from per-match data (broadcast to all matches)
            elif feat_name == "mean_ratio_test":
                X[:, i] = np.mean(ratio_test_values)
            elif feat_name == "mean_n_candidates":
                X[:, i] = np.mean(n_candidates)
            elif feat_name == "n_matches":
                X[:, i] = n_matches
            elif feat_name == "n_inliers":
                X[:, i] = n_matches  # Same as n_matches in this context
            
            # Global geometric features (broadcast to all matches)
            elif geometric_features is not None:
                # Remove 'geom_' prefix if present in model features
                geom_key = feat_name.replace('geom_', '') if feat_name.startswith('geom_') else feat_name
                
                if geom_key in geometric_features:
                    # Broadcast scalar feature to all matches
                    X[:, i] = geometric_features[geom_key]
                else:
                    warnings.warn(f"Geometric feature '{geom_key}' not found, using zeros")
            else:
                # Model expects geometric features but none provided
                warnings.warn(f"Feature '{feat_name}' not available, using zeros")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=50.0, neginf=0.0)
        
        # Predict confidence
        confidence = self.model.predict_proba(X)[:, 1]
        
        return confidence
    
    def confidence_to_pixel_uncertainty(
        self,
        confidence: np.ndarray,
        reprojection_errors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Convert match confidence to pixel-level measurement uncertainty.
        
        Uses the formula: σ = σ_base / sqrt(confidence)
        
        Optionally refines with reprojection error for better gradation.
        
        Args:
            confidence: Match confidence scores [0, 1]
            reprojection_errors: Optional reprojection errors for refinement
            
        Returns:
            sigma: Pixel uncertainty for each match (pixels)
        """
        # Clip confidence to avoid division by zero
        confidence = np.clip(confidence, self.min_confidence, 1.0)
        
        # Base uncertainty from confidence
        sigma = self.sigma_base / np.sqrt(confidence)
        
        # Optional refinement with reprojection error
        if self.use_reprojection_refinement and reprojection_errors is not None:
            # For matches with low reprojection error (<10px), refine uncertainty
            # This adds fine-grained variation within high-confidence matches
            low_error_mask = reprojection_errors < 10.0
            if low_error_mask.any():
                # Scale factor based on reprojection error: [0.5, 1.5]
                error_scale = 0.5 + 0.5 * np.clip(reprojection_errors / 3.0, 0, 1)
                sigma[low_error_mask] *= error_scale[low_error_mask]
        
        return sigma
    
    def compute_pose_covariance(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        pixel_uncertainties: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute 6-DOF pose covariance from measurement uncertainties.
        
        Uses first-order uncertainty propagation:
        Σ_pose = (J^T W J)^{-1}
        
        where:
        - J is the Jacobian of reprojection w.r.t. pose
        - W is the inverse measurement covariance (diagonal from pixel uncertainties)
        
        Args:
            points_3d: 3D points in world coordinates (N x 3)
            points_2d: 2D observations in image (N x 2)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3 x 1)
            pixel_uncertainties: Per-match pixel uncertainties (N,)
            
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
        
        # Build weight matrix
        weights = np.zeros(2 * n_points)
        for i in range(n_points):
            weights[2*i] = 1.0 / (pixel_uncertainties[i] ** 2)
            weights[2*i+1] = 1.0 / (pixel_uncertainties[i] ** 2)
        
        W = np.diag(weights)
        
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
            "mean_pixel_uncertainty": np.mean(pixel_uncertainties),
            "median_pixel_uncertainty": np.median(pixel_uncertainties),
            "std_pixel_uncertainty": np.std(pixel_uncertainties),
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
        camera_matrix: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        best_match_distances: np.ndarray,
        ratio_test_values: np.ndarray,
        n_candidates: np.ndarray,
        reprojection_errors: np.ndarray,
        detector_scores: np.ndarray,
        geometric_features: Optional[Dict] = None,  # NEW
        filter_low_confidence: bool = True
    ) -> Dict:
        """
        End-to-end pose uncertainty estimation pipeline.
        
        Args:
            points_3d: 3D points (N x 3)
            points_2d: 2D observations (N x 2)
            camera_matrix: Camera intrinsics (3 x 3)
            R: Rotation matrix (3 x 3)
            t: Translation vector (3 x 1)
            best_match_distances: Descriptor distances
            ratio_test_values: Lowe's ratio values
            n_candidates: Number of candidates per match
            reprojection_errors: Geometric errors
            detector_scores: XFeat confidence scores
            geometric_features: Dict with global geometric features (NEW)
            filter_low_confidence: Whether to filter out low-confidence matches
            
        Returns:
            result: Dictionary containing:
                - covariance: 6x6 pose covariance
                - confidence: Per-match confidence scores
                - pixel_uncertainties: Per-match uncertainties
                - filtered_indices: Indices of high-confidence matches (if filtered)
                - info: Diagnostic information
        """
        # Step 1: Predict confidence for all matches (with geometric features)
        confidence = self.predict_match_confidence(
            best_match_distances,
            ratio_test_values,
            n_candidates,
            reprojection_errors,
            detector_scores,
            geometric_features=geometric_features  # NEW
        )
        
        # Step 2: Filter low-confidence matches if requested
        if filter_low_confidence:
            mask = confidence >= self.min_confidence
            if mask.sum() < 4:
                warnings.warn(
                    f"Only {mask.sum()} matches above confidence threshold "
                    f"({self.min_confidence}). Using all matches."
                )
                mask = np.ones(len(confidence), dtype=bool)
            
            filtered_indices = np.where(mask)[0]
            confidence_filtered = confidence[mask]
            reprojection_errors_filtered = reprojection_errors[mask]
            points_3d_filtered = points_3d[mask]
            points_2d_filtered = points_2d[mask]
        else:
            filtered_indices = np.arange(len(confidence))
            confidence_filtered = confidence
            reprojection_errors_filtered = reprojection_errors
            points_3d_filtered = points_3d
            points_2d_filtered = points_2d
        
        # Step 3: Convert confidence to pixel uncertainty
        pixel_uncertainties = self.confidence_to_pixel_uncertainty(
            confidence_filtered,
            reprojection_errors_filtered
        )
        
        # Step 4: Compute pose covariance
        covariance, info = self.compute_pose_covariance(
            points_3d_filtered,
            points_2d_filtered,
            camera_matrix,
            R,
            t,
            pixel_uncertainties
        )
        
        return {
            "covariance": covariance,
            "confidence": confidence,
            "pixel_uncertainties": pixel_uncertainties,
            "filtered_indices": filtered_indices,
            "info": info
        }


def print_uncertainty_summary(result: Dict, actual_error: Optional[float] = None):
    """
    Print a human-readable summary of uncertainty estimation results.
    
    Args:
        result: Output from estimate_pose_uncertainty()
        actual_error: Optional ground truth error for validation
    """
    print("\n" + "="*60)
    print("POSE UNCERTAINTY ESTIMATE")
    print("="*60)
    
    info = result['info']
    
    print(f"\nMatch-Level Uncertainty:")
    print(f"  Number of matches:      {info['n_points']}")
    print(f"  Mean confidence:        {np.mean(result['confidence']):.3f}")
    print(f"  Confidence range:       [{np.min(result['confidence']):.3f}, {np.max(result['confidence']):.3f}]")
    print(f"  Mean pixel uncertainty: {info['mean_pixel_uncertainty']:.2f} px")
    print(f"  Median pixel unc.:      {info['median_pixel_uncertainty']:.2f} px")
    
    print(f"\nPose Uncertainty:")
    rot_std = info['rotation_uncertainty']
    trans_std = info['translation_uncertainty']
    print(f"  Rotation std:     [{rot_std[0]:.4f}, {rot_std[1]:.4f}, {rot_std[2]:.4f}] rad")
    print(f"  Translation std:  [{trans_std[0]:.4f}, {trans_std[1]:.4f}, {trans_std[2]:.4f}] m")
    print(f"  Total trans. unc: {np.linalg.norm(trans_std)*1000:.1f} mm")
    
    print(f"\nDiagnostics:")
    print(f"  Mean reprojection error: {info['mean_reprojection_error']:.2f} px")
    print(f"  Information matrix cond: {info['condition_number']:.1e}")
    
    if actual_error is not None:
        trans_unc_total = np.linalg.norm(trans_std)
        print(f"\nValidation:")
        print(f"  Predicted uncertainty: {trans_unc_total*1000:.1f} mm")
        print(f"  Actual error:          {actual_error*1000:.1f} mm")
        print(f"  Ratio (error/unc):     {actual_error/trans_unc_total:.2f}")
    
    print("="*60)