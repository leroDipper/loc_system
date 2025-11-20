import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import glob
import os

def analyze_descriptor_perturbation():
    """
    Compare FP32 vs INT8 descriptors on same images.
    Generate Figure 1 for journal.
    """
    
    # Load models
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # FP32 model
    xfeat_fp32 = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=200)
    xfeat_fp32 = xfeat_fp32.to(device).eval()
    
    # INT8 model
    import onnxruntime as ort
    int8_session = ort.InferenceSession('models/xfeat_640x480_int8.onnx', 
                                        providers=['CPUExecutionProvider'])
    
    # Select test images (use first 50 from FR1)
    image_paths = sorted(glob.glob('resources/tum_fr1/images/*.png'))[:50]
    print(f"Processing {len(image_paths)} images...")
    
    fp32_descriptors = []
    int8_descriptors = []
    
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # FP32 extraction
        with torch.no_grad():
            output_fp32 = xfeat_fp32.detectAndCompute(gray, top_k=200)
        desc_fp32 = output_fp32[0]['descriptors'].cpu().numpy()
        
        # INT8 extraction
        frame_input = gray.astype(np.float32)[np.newaxis, np.newaxis, :, :]
        feats, _, heatmap = int8_session.run(None, {'input': frame_input})
        
        # Get INT8 descriptors (same processing as main code)
        B, C, H, W = feats.shape
        heat_flat = heatmap[0, 0].flatten()
        top_k = min(200, len(heat_flat))
        top_indices = np.argpartition(heat_flat, -top_k)[-top_k:]
        
        feats_flat = feats[0].reshape(64, -1).T
        desc_int8 = feats_flat[top_indices]
        norms = np.linalg.norm(desc_int8, axis=1, keepdims=True)
        desc_int8 = desc_int8 / (norms + 1e-8)
        
        # Store (take first 100 for consistency)
        n = min(100, desc_fp32.shape[0], desc_int8.shape[0])
        fp32_descriptors.append(desc_fp32[:n])
        int8_descriptors.append(desc_int8[:n])
    
    # Flatten all descriptors
    fp32_all = np.vstack(fp32_descriptors)
    int8_all = np.vstack(int8_descriptors)
    
    print(f"Collected {fp32_all.shape[0]} descriptor pairs")
    
    # Compute differences
    differences = np.linalg.norm(fp32_all - int8_all, axis=1)
    fp32_norms = np.linalg.norm(fp32_all, axis=1)
    int8_norms = np.linalg.norm(int8_all, axis=1)
    
    # Statistics
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    max_diff = np.max(differences)
    desc_std = np.std(fp32_all)
    median_diff = np.median(differences)

    # More meaningful relative metrics
    mean_fp32_norm = np.mean(fp32_norms)
    mean_int8_norm = np.mean(int8_norms)
    relative_to_norm = (mean_diff / mean_fp32_norm) * 100

    # Compare to typical inter-descriptor distances (sample 1000 random pairs)
    np.random.seed(42)
    n_samples = min(1000, len(fp32_all))
    idx1 = np.random.choice(len(fp32_all), n_samples, replace=False)
    idx2 = np.random.choice(len(fp32_all), n_samples, replace=False)
    inter_desc_distances = np.linalg.norm(fp32_all[idx1] - fp32_all[idx2], axis=1)
    mean_inter_desc_dist = np.mean(inter_desc_distances)
    relative_to_inter_desc = (mean_diff / mean_inter_desc_dist) * 100


    print("\n" + "="*60)
    print("DESCRIPTOR PERTURBATION STATISTICS")
    print("="*60)
    print(f"Mean L2 difference:           {mean_diff:.4f}")
    print(f"Median L2 difference:         {median_diff:.4f}")
    print(f"Std of differences:           {std_diff:.4f}")
    print(f"Max difference:               {max_diff:.4f}")
    print(f"\nDescriptor norms:")
    print(f"  Mean FP32 norm:             {mean_fp32_norm:.4f}")
    print(f"  Mean INT8 norm:             {mean_int8_norm:.4f}")
    print(f"\nRelative metrics:")
    print(f"  Perturbation / norm:        {relative_to_norm:.1f}%")
    print(f"  Perturbation / inter-desc:  {relative_to_inter_desc:.1f}%")
    print(f"  Mean inter-descriptor dist: {mean_inter_desc_dist:.4f}")
    print("="*60)

    # Save statistics to file
    with open('results/table1_descriptor_stats.txt', 'w') as f:
        f.write("Table 1: Descriptor Statistics\n")
        f.write("="*50 + "\n")
        f.write(f"Mean L2 difference       | {mean_diff:.4f}\n")
        f.write(f"Median L2 difference     | {median_diff:.4f}\n")
        f.write(f"Std of differences       | {std_diff:.4f}\n")
        f.write(f"Max difference           | {max_diff:.4f}\n")
        f.write(f"Mean descriptor norm     | {mean_fp32_norm:.4f}\n")
        f.write(f"Perturbation/norm (%)    | {relative_to_norm:.1f}\n")


    
    
if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    analyze_descriptor_perturbation()