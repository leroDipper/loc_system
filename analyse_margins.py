import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import os
from scipy.spatial.distance import cdist

def analyze_matching_margins():
    """
    Compute matching margins in the SAME space as actual matching pipeline.
    Critical: queries must be scaled to uint8 range [0-255] to match map.
    """
    
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FP32 model
    xfeat_fp32 = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=200)
    xfeat_fp32 = xfeat_fp32.to(device).eval()
    
    # INT8 model
    import onnxruntime as ort
    int8_session = ort.InferenceSession('models/xfeat_640x480_int8.onnx', 
                                        providers=['CPUExecutionProvider'])
    
    # Load map (in uint8 space)
    print("Loading map...")
    data = np.load('resources/tum_fr1/map_databases/tumfr1_map_train.npz')
    map_descriptors = data['descriptors'].astype(np.float32)
    print(f"Map size: {len(map_descriptors)} descriptors")
    print(f"Map descriptor mean norm: {np.mean(np.linalg.norm(map_descriptors, axis=1)):.2f}")
    
    # Test on 20 query images
    image_paths = sorted(glob.glob('resources/tum_fr1/images/*.png'))[500:520]
    print(f"Processing {len(image_paths)} query images...\n")
    
    fp32_margins = []
    int8_margins = []
    fp32_match_success = []
    int8_match_success = []
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ==================== FP32 EXTRACTION ====================
        with torch.no_grad():
            output_fp32 = xfeat_fp32.detectAndCompute(gray, top_k=100)
        query_fp32 = output_fp32[0]['descriptors'].cpu().numpy()
        
        # CRITICAL: Scale to uint8 space [0-255] to match map
        query_fp32 = np.clip((query_fp32 + 0.5) * 255.0, 0, 255).astype(np.float32)
        
        # ==================== INT8 EXTRACTION ====================
        frame_input = gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        feats, _, heatmap = int8_session.run(None, {'input': frame_input})
        
        B, C, H, W = feats.shape
        heat_flat = heatmap[0, 0].flatten()
        top_k = min(100, len(heat_flat))
        top_indices = np.argpartition(heat_flat, -top_k)[-top_k:]
        
        feats_flat = feats[0].reshape(64, -1).T
        query_int8 = feats_flat[top_indices]
        norms = np.linalg.norm(query_int8, axis=1, keepdims=True)
        query_int8 = query_int8 / (norms + 1e-8)
        
        # CRITICAL: Scale to uint8 space [0-255] to match map
        query_int8 = np.clip((query_int8 + 0.5) * 255.0, 0, 255).astype(np.float32)
        
        # Take minimum size for fair comparison
        n_queries = min(query_fp32.shape[0], query_int8.shape[0])
        query_fp32 = query_fp32[:n_queries]
        query_int8 = query_int8[:n_queries]
        
        # ==================== COMPUTE MARGINS ====================
        # FP32 margins
        distances_fp32 = cdist(query_fp32, map_descriptors, metric='euclidean')
        for q_idx in range(n_queries):
            dists = distances_fp32[q_idx]
            sorted_dists = np.sort(dists)
            best_dist = sorted_dists[0]
            second_best_dist = sorted_dists[1]
            margin = second_best_dist - best_dist
            fp32_margins.append(margin)
            fp32_match_success.append(best_dist < 0.8 * second_best_dist)
        
        # INT8 margins
        distances_int8 = cdist(query_int8, map_descriptors, metric='euclidean')
        for q_idx in range(n_queries):
            dists = distances_int8[q_idx]
            sorted_dists = np.sort(dists)
            best_dist = sorted_dists[0]
            second_best_dist = sorted_dists[1]
            margin = second_best_dist - best_dist
            int8_margins.append(margin)
            int8_match_success.append(best_dist < 0.8 * second_best_dist)
    
    fp32_margins = np.array(fp32_margins)
    int8_margins = np.array(int8_margins)
    fp32_match_success = np.array(fp32_match_success)
    int8_match_success = np.array(int8_match_success)
    
    print("="*60)
    print("MARGIN ANALYSIS (in uint8 descriptor space)")
    print("="*60)
    print(f"Total descriptors analyzed: {len(fp32_margins)}")
    print(f"\nFP32 margins:")
    print(f"  Mean:   {np.mean(fp32_margins):.2f}")
    print(f"  Median: {np.median(fp32_margins):.2f}")
    print(f"  Std:    {np.std(fp32_margins):.2f}")
    print(f"  Min:    {np.min(fp32_margins):.2f}")
    print(f"  Passes Lowe's ratio: {np.sum(fp32_match_success)}/{len(fp32_match_success)} ({100*np.mean(fp32_match_success):.1f}%)")
    
    print(f"\nINT8 margins:")
    print(f"  Mean:   {np.mean(int8_margins):.2f}")
    print(f"  Median: {np.median(int8_margins):.2f}")
    print(f"  Std:    {np.std(int8_margins):.2f}")
    print(f"  Min:    {np.min(int8_margins):.2f}")
    print(f"  Passes Lowe's ratio: {np.sum(int8_match_success)}/{len(int8_match_success)} ({100*np.mean(int8_match_success):.1f}%)")
    
    margin_degradation = np.mean(fp32_margins) - np.mean(int8_margins)
    print(f"\nMargin degradation (FP32 - INT8): {margin_degradation:.2f}")
    print(f"Relative degradation: {100*margin_degradation/np.mean(fp32_margins):.1f}%")
    
    # Count how many margins become negative (match flips)
    negative_int8 = np.sum(int8_margins < 0)
    print(f"\nNegative margins (match failures):")
    print(f"  FP32: {np.sum(fp32_margins < 0)}/{len(fp32_margins)}")
    print(f"  INT8: {negative_int8}/{len(int8_margins)} ({100*negative_int8/len(int8_margins):.1f}%)")
    print("="*60)
    
    # ==================== PLOT ====================
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel (a): Margin distributions
    ax = axes[0, 0]
    ax.hist(fp32_margins, bins=50, alpha=0.6, label='FP32', color='blue', edgecolor='black')
    ax.hist(int8_margins, bins=50, alpha=0.6, label='INT8', color='orange', edgecolor='black')
    ax.axvline(np.mean(fp32_margins), color='blue', linestyle='--', linewidth=2)
    ax.axvline(np.mean(int8_margins), color='orange', linestyle='--', linewidth=2)
    ax.axvline(0, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Zero margin')
    ax.set_xlabel('Margin (2nd_best_dist - best_dist)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(a) Matching Margin Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Panel (b): CDF of margins
    ax = axes[0, 1]
    sorted_fp32 = np.sort(fp32_margins)
    sorted_int8 = np.sort(int8_margins)
    cdf_fp32 = np.arange(1, len(sorted_fp32) + 1) / len(sorted_fp32)
    cdf_int8 = np.arange(1, len(sorted_int8) + 1) / len(sorted_int8)
    ax.plot(sorted_fp32, cdf_fp32, label='FP32', linewidth=2, color='blue')
    ax.plot(sorted_int8, cdf_int8, label='INT8', linewidth=2, color='orange')
    ax.axvline(0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Zero margin')
    ax.set_xlabel('Margin', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('(b) Cumulative Margin Distribution', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel (c): Margin change per descriptor
    ax = axes[1, 0]
    margin_change = int8_margins - fp32_margins
    ax.hist(margin_change, bins=50, color='purple', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(margin_change), color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(margin_change):.2f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('Margin Change (INT8 - FP32)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('(c) Per-Descriptor Margin Degradation', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel (d): Scatter: FP32 margin vs INT8 margin
    ax = axes[1, 1]
    sample_size = min(2000, len(fp32_margins))
    sample_idx = np.random.choice(len(fp32_margins), sample_size, replace=False)
    ax.scatter(fp32_margins[sample_idx], int8_margins[sample_idx], 
               alpha=0.3, s=10, color='steelblue')
    max_val = max(fp32_margins.max(), int8_margins.max())
    min_val = min(fp32_margins.min(), int8_margins.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x')
    ax.axhline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xlabel('FP32 Margin', fontsize=11)
    ax.set_ylabel('INT8 Margin', fontsize=11)
    ax.set_title('(d) Margin Preservation', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figure_margin_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure_margin_analysis.pdf', bbox_inches='tight')
    print("\n✓ Saved: results/figure_margin_analysis.png")
    print("✓ Saved: results/figure_margin_analysis.pdf")
    
    # Save stats
    with open('results/margin_stats.txt', 'w') as f:
        f.write("Margin Analysis Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"FP32 mean margin:        {np.mean(fp32_margins):.2f}\n")
        f.write(f"INT8 mean margin:        {np.mean(int8_margins):.2f}\n")
        f.write(f"Margin degradation:      {margin_degradation:.2f}\n")
        f.write(f"Relative degradation:    {100*margin_degradation/np.mean(fp32_margins):.1f}%\n")
        f.write(f"FP32 Lowe's pass rate:   {100*np.mean(fp32_match_success):.1f}%\n")
        f.write(f"INT8 Lowe's pass rate:   {100*np.mean(int8_match_success):.1f}%\n")
        f.write(f"INT8 negative margins:   {100*negative_int8/len(int8_margins):.1f}%\n")
    
    print("✓ Saved: results/margin_stats.txt")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    analyze_matching_margins()