"""
Test the pose uncertainty module with synthetic data.

This verifies that the uncertainty estimation pipeline works correctly.
"""

import numpy as np
import sys
from loc_modules.pose_uncertainty import PoseUncertaintyEstimator, print_uncertainty_summary


def generate_synthetic_data(n_points=50):
    """Generate synthetic matches for testing."""
    np.random.seed(42)
    
    # Generate 3D points
    points_3d = np.random.randn(n_points, 3) * 2.0
    points_3d[:, 2] += 5.0  # Offset in Z to be in front of camera
    
    # Camera parameters
    camera_matrix = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ])
    
    # Ground truth pose
    R = np.eye(3)  # Identity rotation
    t = np.array([0.1, 0.0, 0.0])  # Small translation
    
    # Project points
    points_camera = (R @ points_3d.T).T + t.reshape(1, 3)
    points_proj_homo = camera_matrix @ points_camera.T
    points_2d = (points_proj_homo[:2, :] / points_proj_homo[2, :]).T
    
    # Add small noise
    points_2d += np.random.randn(n_points, 2) * 0.5
    
    # Generate match features
    # Good matches: low distance, low ratio, high confidence
    best_match_distances = np.random.uniform(0.1, 0.4, n_points)
    ratio_test_values = np.random.uniform(0.4, 0.6, n_points)
    n_candidates = np.ones(n_points)
    detector_scores = np.random.uniform(0.2, 0.3, n_points)
    
    # Add some variation in quality
    # Make 20% of matches slightly worse
    n_worse = n_points // 5
    best_match_distances[:n_worse] += 0.2
    ratio_test_values[:n_worse] += 0.1
    
    # Compute reprojection errors
    reprojection_errors = np.linalg.norm(points_2d - (points_proj_homo[:2, :] / points_proj_homo[2, :]).T, axis=1)
    
    return {
        'points_3d': points_3d,
        'points_2d': points_2d,
        'camera_matrix': camera_matrix,
        'R': R,
        't': t,
        'best_match_distances': best_match_distances,
        'ratio_test_values': ratio_test_values,
        'n_candidates': n_candidates,
        'reprojection_errors': reprojection_errors,
        'detector_scores': detector_scores
    }


def main():
    print("="*60)
    print("TESTING POSE UNCERTAINTY MODULE")
    print("="*60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    data = generate_synthetic_data(n_points=50)
    print(f"   ✓ Generated {len(data['points_3d'])} synthetic matches")
    
    # Initialize estimator
    print("\n2. Loading trained model...")
    try:
        estimator = PoseUncertaintyEstimator(
            model_path="results/match_confidence_model.joblib",
            sigma_base=1.0,
            min_confidence=0.3,
            use_reprojection_refinement=True
        )
    except FileNotFoundError:
        print("   ✗ Model not found! Please train it first:")
        print("     python train_match_confidence_v2.py")
        return
    
    # Estimate uncertainty
    print("\n3. Estimating pose uncertainty...")
    result = estimator.estimate_pose_uncertainty(
        points_3d=data['points_3d'],
        points_2d=data['points_2d'],
        camera_matrix=data['camera_matrix'],
        R=data['R'],
        t=data['t'],
        best_match_distances=data['best_match_distances'],
        ratio_test_values=data['ratio_test_values'],
        n_candidates=data['n_candidates'],
        reprojection_errors=data['reprojection_errors'],
        detector_scores=data['detector_scores'],
        filter_low_confidence=True
    )
    
    # Print results
    print_uncertainty_summary(result)
    
    # Analyze covariance
    print("\n4. Analyzing pose covariance matrix...")
    cov = result['covariance']
    print(f"\nCovariance matrix shape: {cov.shape}")
    print(f"\nRotation covariance (top-left 3x3):")
    print(cov[:3, :3])
    print(f"\nTranslation covariance (bottom-right 3x3):")
    print(cov[3:, 3:])
    
    # Check if covariance is positive definite
    eigenvalues = np.linalg.eigvals(cov)
    if np.all(eigenvalues > 0):
        print(f"\n✓ Covariance is positive definite (all eigenvalues > 0)")
    else:
        print(f"\n✗ Warning: Covariance has negative eigenvalues!")
    
    print(f"\nEigenvalues: {eigenvalues}")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()