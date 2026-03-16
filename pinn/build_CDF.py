import numpy as np
import joblib

cdf_errors = joblib.load('results/empirical_cdf.joblib')

def score_to_error(score, interval=10):
    """
    Convert confidence score (1=certain, 0=uncertain) to metre-level error estimate.
    
    Args:
        score: float in [0, 1], 1=certain
        interval: percentile interval for confidence bounds (default ±10)
    
    Returns:
        estimate, lower, upper in metres
    """
    q = (1.0 - score) * 100
    estimate = np.percentile(cdf_errors, q)
    lower = np.percentile(cdf_errors, max(0, q - interval))
    upper = np.percentile(cdf_errors, min(100, q + interval))
    return estimate, lower, upper

if __name__ == '__main__':
    print(f"CDF built from {len(cdf_errors)} frames")
    print(f"Error range: {cdf_errors.min()*100:.1f}cm — {cdf_errors.max()*100:.1f}cm")
    print("\nExample lookups:")
    for score in [0.20, 0.40, 0.60, 0.80, 0.90, 0.95]:
        est, lo, hi = score_to_error(score)
        print(f"  score={score:.2f} → {est*100:.1f}cm [{lo*100:.1f} - {hi*100:.1f}cm]")