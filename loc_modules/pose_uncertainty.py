"""
Pose Uncertainty Estimation Module

Converts match-level confidence scores into pose covariance using:
1. ML model: match features → confidence p
2. Uncertainty scaling: p → pixel uncertainty σ
3. Covariance propagation: σ → pose covariance Σ

Based on the factor graph formulation where measurement uncertainty
is derived from learned match confidence.
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
                f"Train it first using train_match_confidence_v2.py"
            )
    
    def predict_match_confidence(
        self,
        best_match_distances: np.ndarray,
        ratio_test_values: np.ndarray,
        n_candidates: np.ndarray,
        reprojection_errors: np.ndarray,
        detector_scores: np.ndarray
    ) -> np.ndarray:
        """
        Predict confidence for each match using the ML model.
        
        Args:
            best_match_distances: Descriptor distance to best match
            ratio_test_values: Lowe's ratio test values
            n_candidates: Number of candidate matches per feature
            reprojection_errors: Geometric reprojection errors (pixels)
            detector_scores: XFeat detector confidence scores
            
        Returns:
            confidence: Array of confidence scores [0, 1] for each match
        """
        # Build feature matrix
        n_matches = len(best_match_distances)
        X = np.zeros((n_matches, len(self.feature_names)))
        
        for i, feat_name in enumerate(self.feature_names):
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
            else:
                warnings.warn(f"Unknown feature: {feat_name}, using zeros")
        
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
        # Each point contributes 2 rows (u, v) with respect to 6 DOF pose
        J = np.zeros((2 * n_points, 6))
        
        for i in range(n_points):
            X, Y, Z = points_camera[i]
            Z2 = Z * Z
            
            # Jacobian of projection w.r.t. camera-frame point
            # u = fx * X/Z + cx, v = fy * Y/Z + cy
            du_dX = fx / Z
            du_dY = 0
            du_dZ = -fx * X / Z2
            
            dv_dX = 0
            dv_dY = fy / Z
            dv_dZ = -fy * Y / Z2
            
            # Jacobian w.r.t. rotation (using small angle approximation)
            # Rotation Jacobian: δX = [X]_× δω where [X]_× is skew-symmetric
            # This gives: dX/dω = [0, Z, -Y; -Z, 0, X; Y, -X, 0]^T
            
            # For rotation (ωx, ωy, ωz):
            J[2*i, 0] = du_dX * 0      + du_dY * Z      + du_dZ * (-Y)  # ωx
            J[2*i, 1] = du_dX * (-Z)   + du_dY * 0      + du_dZ * X     # ωy
            J[2*i, 2] = du_dX * Y      + du_dY * (-X)   + du_dZ * 0     # ωz
            
            J[2*i+1, 0] = dv_dX * 0    + dv_dY * Z      + dv_dZ * (-Y)  # ωx
            J[2*i+1, 1] = dv_dX * (-Z) + dv_dY * 0      + dv_dZ * X     # ωy
            J[2*i+1, 2] = dv_dX * Y    + dv_dY * (-X)   + dv_dZ * 0     # ωz
            
            # For translation (tx, ty, tz):
            J[2*i, 3]   = du_dX  # tx
            J[2*i, 4]   = du_dY  # ty
            J[2*i, 5]   = du_dZ  # tz
            
            J[2*i+1, 3] = dv_dX  # tx
            J[2*i+1, 4] = dv_dY  # ty
            J[2*i+1, 5] = dv_dZ  # tz
        
        # Build weight matrix (inverse covariance)
        # W = diag(1/σ₁², 1/σ₁², 1/σ₂², 1/σ₂², ...)
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
        # Reprojection for verification
        points_proj_homo = camera_matrix @ points_camera.T
        points_proj = (points_proj_homo[:2, :] / points_proj_homo[2, :]).T
        reprojection_errors = np.linalg.norm(points_2d - points_proj, axis=1)
        
        info = {
            "n_points": n_points,
            "mean_pixel_uncertainty": np.mean(pixel_uncertainties),
            "median_pixel_uncertainty": np.median(pixel_uncertainties),
            "mean_reprojection_error": np.mean(reprojection_errors),
            "condition_number": np.linalg.cond(H),
            "rotation_uncertainty": np.sqrt(np.diag(covariance[:3, :3])),  # Angular uncertainty (radians)
            "translation_uncertainty": np.sqrt(np.diag(covariance[3:, 3:])),  # Translation uncertainty (meters)
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
            filter_low_confidence: Whether to filter out low-confidence matches
            
        Returns:
            result: Dictionary containing:
                - covariance: 6x6 pose covariance
                - confidence: Per-match confidence scores
                - pixel_uncertainties: Per-match uncertainties
                - filtered_indices: Indices of high-confidence matches (if filtered)
                - info: Diagnostic information
        """
        # Step 1: Predict confidence for all matches
        confidence = self.predict_match_confidence(
            best_match_distances,
            ratio_test_values,
            n_candidates,
            reprojection_errors,
            detector_scores
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
        
        # Package results
        result = {
            "covariance": covariance,
            "confidence": confidence,
            "confidence_filtered": confidence_filtered,
            "pixel_uncertainties": pixel_uncertainties,
            "filtered_indices": filtered_indices,
            "n_filtered": len(filtered_indices),
            "n_total": len(confidence),
            "info": info
        }
        
        return result


def print_uncertainty_summary(result: Dict):
    """Print a human-readable summary of uncertainty estimation results."""
    print("\n" + "="*60)
    print("POSE UNCERTAINTY SUMMARY")
    print("="*60)
    
    print(f"\nMatches:")
    print(f"  Total:    {result['n_total']}")
    print(f"  Filtered: {result['n_filtered']} "
          f"({result['n_filtered']/result['n_total']*100:.1f}%)")
    
    print(f"\nConfidence Statistics:")
    conf = result['confidence_filtered']
    print(f"  Min:    {conf.min():.3f}")
    print(f"  25th:   {np.percentile(conf, 25):.3f}")
    print(f"  Median: {np.median(conf):.3f}")
    print(f"  75th:   {np.percentile(conf, 75):.3f}")
    print(f"  Max:    {conf.max():.3f}")
    
    print(f"\nPixel Uncertainty:")
    sigma = result['pixel_uncertainties']
    print(f"  Mean:   {sigma.mean():.2f} px")
    print(f"  Median: {np.median(sigma):.2f} px")
    print(f"  Std:    {sigma.std():.2f} px")
    
    info = result['info']
    print(f"\nPose Uncertainty:")
    print(f"  Rotation (std):    [{info['rotation_uncertainty'][0]:.4f}, "
          f"{info['rotation_uncertainty'][1]:.4f}, "
          f"{info['rotation_uncertainty'][2]:.4f}] rad")
    print(f"  Translation (std): [{info['translation_uncertainty'][0]:.4f}, "
          f"{info['translation_uncertainty'][1]:.4f}, "
          f"{info['translation_uncertainty'][2]:.4f}] m")
    
    print(f"\nDiagnostics:")
    print(f"  Mean reprojection error: {info['mean_reprojection_error']:.2f} px")
    print(f"  Condition number:        {info['condition_number']:.2e}")
    
    print("="*60)