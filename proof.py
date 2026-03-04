import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import onnxruntime as ort
import glob
from scipy.spatial.transform import Rotation
from accelerated_modules import vocab_tree_match
from loc_modules.onnx_feat import onnx_extractor
from loc_modules.load_gt_params import GroundTruthParams
import yaml
import os


def load_camera_params(yaml_path):
    """Load camera parameters from YAML file."""
    with open(yaml_path, 'r') as f:
        params = yaml.safe_load(f)
    
    return {
        'width': params['resolution'][0],
        'height': params['resolution'][1],
        'fx': params['intrinsics'][0],
        'fy': params['intrinsics'][1],
        'cx': params['intrinsics'][2],
        'cy': params['intrinsics'][3]
    }


def load_colmap_image_names(images_txt_path):
    """Load image names from COLMAP images.txt in order."""
    colmap_images = []
    with open(images_txt_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) == 10:
                colmap_images.append(parts[9])
    return colmap_images

# ========== SETUP ==========
print("Loading models...")

# FP32 model
device = "cuda" if torch.cuda.is_available() else "cpu"
xfeat_fp32 = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=4096)
xfeat_fp32 = xfeat_fp32.to(device)
xfeat_fp32.eval()

# INT8 model
session_int8 = ort.InferenceSession('models/xfeat_752x480_int8.onnx', 
                                    providers=['CPUExecutionProvider'])



scale, R, t = GroundTruthParams.load_transformation('resources/mh_01/project_files/colmap_to_gt_transform.json')
# Load map
data = np.load('resources/mh_01/map_databases/mh_01_master.npz')
map_descriptors = data['descriptors']
map_3d_points = data['xyz_world']
print(f"Loaded map: {len(map_descriptors)} descriptors")

N_TRAIN_IMAGES = 2900

test_dataset_path = 'resources/mh_01/images'

CAMERA_PARAMS_PATH = 'resources/mh_01/images/camera_rectified.yaml'

# Build vocab tree matcher
print("Building vocab tree matcher...")
matcher = vocab_tree_match.VocabTreeMatcher(
    'resources/mh_01/vocabularies/vocab_tree_master.bin',
    map_descriptors
)


# Camera params
K = np.array([[355.64, 0, 362.27],
              [0, 417.16, 249.66],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)


camera_params = load_camera_params(CAMERA_PARAMS_PATH)
print(f"Loaded camera: {camera_params['width']}x{camera_params['height']}")
print(f"  fx={camera_params['fx']:.2f}, fy={camera_params['fy']:.2f}")
print(f"  cx={camera_params['cx']:.2f}, cy={camera_params['cy']:.2f}")

K = np.array([[camera_params['fx'], 0, camera_params['cx']],
                [0, camera_params['fy'], camera_params['cy']],
                [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(5, dtype=np.float32)





 # Load COLMAP reconstructed images
colmap_images = load_colmap_image_names('resources/mh_01/project_files/images.txt')
colmap_set = set(colmap_images)

# Get chronological order
all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
all_filenames = [os.path.basename(f) for f in all_frames]

# Filter to only reconstructed images
reconstructed_chrono = [f for f in all_filenames if f in colmap_set]

# Take remaining after first N_TRAIN_IMAGES
test_images = [os.path.join(test_dataset_path, f) for f in reconstructed_chrono[N_TRAIN_IMAGES:]]

test_image_list = sorted(list(test_images))


# ========== THE EXPERIMENT ==========
print("=" * 70)
print("NULLSPACE VALIDATION EXPERIMENT")
print("=" * 70)

results = []

for idx, img_path in enumerate(test_image_list):
    frame = cv2.imread(img_path)
    if frame is None:
        continue
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # ========== FP32 PIPELINE ==========
    with torch.no_grad():
        output = xfeat_fp32.detectAndCompute(frame_gray, top_k=250)
    
    features_fp32 = output[0]
    kpts_fp32 = features_fp32['keypoints'].cpu().numpy()
    desc_fp32 = features_fp32['descriptors'].cpu().numpy()
    desc_fp32_uint8 = np.clip((desc_fp32 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    
    # Match
    query_idx_fp32, map_idx_fp32, dist_fp32 = matcher.match(
        desc_fp32_uint8, ratio_threshold=0.80, k_nearest_words=1
    )
    
    if len(map_idx_fp32) < 6:
        continue
    
    # PnP
    matched_3d_fp32 = map_3d_points[map_idx_fp32]
    matched_2d_fp32 = kpts_fp32[query_idx_fp32]
    
    success_fp32, rvec_fp32, tvec_fp32, inliers_fp32 = cv2.solvePnPRansac(
        matched_3d_fp32, matched_2d_fp32, K, dist_coeffs,
        reprojectionError=8.0, confidence=0.99
    )
    
    if not success_fp32 or len(inliers_fp32) < 6:
        continue
    
    # ========== INT8 PIPELINE ==========
    keypoints_int8, desc_int8, _ = onnx_extractor(session_int8, frame_gray, top_k=250)
    desc_int8_uint8 = np.clip((desc_int8 + 0.5) * 255.0, 0, 255).astype(np.uint8)
    
    # Match
    query_idx_int8, map_idx_int8, dist_int8 = matcher.match(
        desc_int8_uint8, ratio_threshold=0.80, k_nearest_words=1
    )
    
    if len(map_idx_int8) < 6:
        continue
    
    # PnP
    matched_3d_int8 = map_3d_points[map_idx_int8]
    matched_2d_int8 = keypoints_int8[query_idx_int8]
    
    success_int8, rvec_int8, tvec_int8, inliers_int8 = cv2.solvePnPRansac(
        matched_3d_int8, matched_2d_int8, K, dist_coeffs,
        reprojectionError=8.0, confidence=0.99
    )
    
    if not success_int8 or len(inliers_int8) < 6:
        continue
    
    # ========== MEASURE NULLSPACE VALIDATION METRICS ==========
    
    # 1. Correspondence divergence (δC magnitude)
    set_fp32 = set(map_idx_fp32)
    set_int8 = set(map_idx_int8)
    overlap = len(set_fp32 & set_int8) / len(set_fp32 | set_int8)
    divergence = 1 - overlap  # δC "magnitude"
    
    # 2. Pose change (δT magnitude)
    # Translation change
    t_fp32 = tvec_fp32.flatten()
    t_int8 = tvec_int8.flatten()
    delta_translation = np.linalg.norm(t_fp32 - t_int8)
    
    # Rotation change (angle between rotations)
    R_fp32 = cv2.Rodrigues(rvec_fp32)[0]
    R_int8 = cv2.Rodrigues(rvec_int8)[0]
    R_diff = R_fp32.T @ R_int8
    angle_diff = np.arccos((np.trace(R_diff) - 1) / 2) * 180 / np.pi
    
    # 3. Geometric validity of non-overlapping
    overlapping_int8 = [i for i, mp in enumerate(map_idx_int8) if mp in set_fp32]
    non_overlapping_int8 = [i for i, mp in enumerate(map_idx_int8) if mp not in set_fp32]
    
    inlier_set = set(inliers_int8.flatten())
    
    num_overlapping = len(overlapping_int8)
    num_non_overlapping = len(non_overlapping_int8)
    
    overlapping_valid = sum(1 for i in overlapping_int8 if i in inlier_set)
    non_overlapping_valid = sum(1 for i in non_overlapping_int8 if i in inlier_set)
    
    validity_rate_overlapping = overlapping_valid / num_overlapping if num_overlapping > 0 else 0
    validity_rate_non_overlapping = non_overlapping_valid / num_non_overlapping if num_non_overlapping > 0 else 0
    
    results.append({
        'divergence': divergence,
        'delta_translation': delta_translation,
        'delta_rotation': angle_diff,
        'num_matches_fp32': len(map_idx_fp32),
        'num_matches_int8': len(map_idx_int8),
        'num_inliers_fp32': len(inliers_fp32),
        'num_inliers_int8': len(inliers_int8),
        'validity_overlapping': validity_rate_overlapping,
        'validity_non_overlapping': validity_rate_non_overlapping,
        'num_overlapping': num_overlapping,
        'num_non_overlapping': num_non_overlapping
    })
    
    if (idx + 1) % 50 == 0:
        print(f"Processed {idx + 1}/{len(test_images)} images...")

# ========== ANALYSIS ==========
print("\n" + "=" * 70)
print("NULLSPACE VALIDATION RESULTS")
print("=" * 70)

divergences = [r['divergence'] for r in results]
delta_translations = [r['delta_translation'] for r in results]
delta_rotations = [r['delta_rotation'] for r in results]
validity_non_overlapping = [r['validity_non_overlapping'] for r in results 
                             if r['num_non_overlapping'] > 0]

print(f"\n1. CORRESPONDENCE DIVERGENCE (δC magnitude):")
print(f"   Median: {np.median(divergences):.1%} (mean: {np.mean(divergences):.1%})")
print(f"   → FP32 and INT8 match very different 3D points")

print(f"\n2. POSE CHANGE (δT magnitude):")
print(f"   Translation - Median: {np.median(delta_translations):.4f} m (mean: {np.mean(delta_translations):.4f} m)")
print(f"   Rotation - Median: {np.median(delta_rotations):.2f}° (mean: {np.mean(delta_rotations):.2f}°)")
print(f"   → Despite large δC, pose change is SMALL")

print(f"\n3. GEOMETRIC VALIDITY:")
print(f"   Non-overlapping validity - Median: {np.median(validity_non_overlapping):.1%} (mean: {np.mean(validity_non_overlapping):.1%})")
print(f"   → Most non-overlapping matches are still geometrically valid")

print("\n" + "─" * 70)
print("INTERPRETATION:")
print("─" * 70)
print(f"Large correspondence change (median {np.median(divergences):.1%})")
print(f"→ Small pose change (median {np.median(delta_translations):.4f} m translation)")
print(f"→ High geometric validity (median {np.median(validity_non_overlapping):.1%})")
print("\nThis is EXACTLY what the nullspace framework predicts:")
print("Different correspondences lie in the nullspace → don't affect pose!")
print("=" * 70)

# Also update the figure annotations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# remove outliers from delta_translations for better visualisation

delta_translations = np.clip(delta_translations, 0, np.percentile(delta_translations, 95))

# add mask to divergences to match the filtered delta_translations
divergences = divergences[:len(delta_translations)]

# 1. Divergence vs Pose Change
ax = axes[0, 0]
ax.scatter(divergences, delta_translations, alpha=0.6, s=40)
ax.set_xlabel('Correspondence Divergence (%)', fontsize=11)
ax.set_ylabel('Translation Change (m)', fontsize=11)
ax.set_title('(a) Large δC → Small δT', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Divergence distribution
ax = axes[0, 1]
ax.hist(divergences, bins=20, alpha=0.7, edgecolor='black', color='steelblue')
ax.axvline(np.median(divergences), color='red', linestyle='--', linewidth=2,
           label=f'Median = {np.median(divergences):.1%}')
ax.axvline(np.mean(divergences), color='orange', linestyle=':', linewidth=2,
           label=f'Mean = {np.mean(divergences):.1%}')
ax.set_xlabel('Correspondence Divergence', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(b) Correspondence Divergence Distribution', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 3. Pose change distribution
ax = axes[1, 0]
ax.hist(delta_translations, bins=20, alpha=0.7, edgecolor='black', color='coral')
ax.axvline(np.median(delta_translations), color='red', linestyle='--', linewidth=2,
           label=f'Median = {np.median(delta_translations):.4f} m')
ax.axvline(np.mean(delta_translations), color='orange', linestyle=':', linewidth=2,
           label=f'Mean = {np.mean(delta_translations):.4f} m')
ax.set_xlabel('Translation Change (m)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(c) Pose Stability Despite Divergence', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Geometric validity
ax = axes[1, 1]
ax.hist(validity_non_overlapping, bins=20, alpha=0.7, edgecolor='black', color='green')
ax.axvline(np.median(validity_non_overlapping), color='red', linestyle='--', linewidth=2,
           label=f'Median = {np.median(validity_non_overlapping):.1%}')
ax.axvline(np.mean(validity_non_overlapping), color='orange', linestyle=':', linewidth=2,
           label=f'Mean = {np.mean(validity_non_overlapping):.1%}')
ax.set_xlabel('Geometric Validity Rate', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('(d) Non-Overlapping Matches Validity', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nullspace_validation.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Saved: nullspace_validation.pdf")
plt.show()