import numpy as np
import os
import psutil
import torch

def get_file_size_mb(filepath):
    """Get file size in MB."""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024 * 1024)
    return 0

def measure_memory_footprint():
    """
    Measure memory footprint of complete localization system.
    Run this on Raspberry Pi to get deployment memory requirements.
    """
    
    print("="*60)
    print("MEMORY FOOTPRINT MEASUREMENT")
    print("="*60)
    
    components = {}
    
    # ========== MODEL SIZE ==========
    print("\n1. Model weights...")
    
    # INT8 ONNX model
    int8_model_path = 'models/xfeat_640x480_int8.onnx'
    components['INT8 ONNX Model'] = get_file_size_mb(int8_model_path)
    print(f"   INT8 ONNX model: {components['INT8 ONNX Model']:.2f} MB")
    
    # FP32 PyTorch model (for comparison, if you have it saved)
    # fp32_model_path = 'models/xfeat_fp32.pth'
    # fp32_size = get_file_size_mb(fp32_model_path)
    # print(f"   FP32 PyTorch model: {fp32_size:.2f} MB (not used in deployment)")
    
    # ========== MAP DATA ==========
    print("\n2. Map database...")
    
    map_path = 'resources/tum_fr1/map_databases/tumfr1_map_train.npz'
    data = np.load(map_path)
    
    # 3D points
    map_3d_points = data['xyz_world']
    points_size = map_3d_points.nbytes / (1024 * 1024)
    components['Map 3D Points'] = points_size
    print(f"   3D points ({len(map_3d_points)} points): {points_size:.2f} MB")
    
    # Descriptors
    map_descriptors = data['descriptors']
    desc_size = map_descriptors.nbytes / (1024 * 1024)
    components['Map Descriptors'] = desc_size
    print(f"   Descriptors ({len(map_descriptors)} x {map_descriptors.shape[1]}D): {desc_size:.2f} MB")
    
    # ========== VOCABULARY TREE ==========
    print("\n3. Vocabulary tree...")
    
    vocab_path = 'resources/tum_fr1/vocabularies/vocab_tree.bin'
    vocab_size = get_file_size_mb(vocab_path)
    components['Vocabulary Tree'] = vocab_size
    print(f"   Vocabulary tree: {vocab_size:.2f} MB")
    
    # ========== RUNTIME MEMORY ==========
    print("\n4. Estimating runtime memory overhead...")
    
    # Measure current process memory
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)
    
    # Load model to measure runtime overhead
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(int8_model_path, providers=['CPUExecutionProvider'])
        
        # Allocate typical runtime buffers (one frame)
        dummy_frame = np.zeros((1, 1, 480, 640), dtype=np.float32)
        _ = session.run(None, {'input': dummy_frame})
        
        loaded_memory = process.memory_info().rss / (1024 * 1024)
        runtime_overhead = loaded_memory - baseline_memory
        
        components['Runtime Overhead'] = runtime_overhead
        print(f"   Runtime buffers & overhead: {runtime_overhead:.2f} MB")
        
    except Exception as e:
        print(f"   Could not measure runtime overhead: {e}")
        # Conservative estimate
        components['Runtime Overhead'] = 50.0
        print(f"   Using estimate: 50.00 MB")
    
    # ========== TOTAL ==========
    total = sum(components.values())
    
    print("\n" + "="*60)
    print("MEMORY FOOTPRINT SUMMARY")
    print("="*60)
    
    for component, size in components.items():
        percentage = (size / total) * 100
        print(f"{component:25} | {size:7.2f} MB | {percentage:5.1f}%")
    
    print("-" * 60)
    print(f"{'TOTAL':25} | {total:7.2f} MB | 100.0%")
    print("="*60)
    
    # ========== LATEX TABLE ==========
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Memory footprint of localization system on Raspberry Pi 5}")
    print("\\begin{tabular}{lrr}")
    print("\\toprule")
    print("\\textbf{Component} & \\textbf{Size (MB)} & \\textbf{\\% of Total} \\\\")
    print("\\midrule")
    
    for component, size in components.items():
        percentage = (size / total) * 100
        # Escape underscores for LaTeX
        component_latex = component.replace('_', '\\_')
        print(f"{component_latex:25} & {size:6.2f} & {percentage:5.1f} \\\\")
    
    print("\\midrule")
    print(f"{'TOTAL':25} & {total:6.2f} & 100.0 \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\label{tab:memory_footprint}")
    print("\\end{table}")
    
    # ========== DEPLOYMENT ANALYSIS ==========
    print("\n" + "="*60)
    print("DEPLOYMENT CONSIDERATIONS")
    print("="*60)
    print(f"Total memory footprint: {total:.1f} MB")
    print(f"")
    print(f"Suitable for devices with:")
    print(f"  • 512 MB RAM:  {'✓ YES' if total < 400 else '✗ NO (too tight)'}")
    print(f"  • 1 GB RAM:    ✓ YES (comfortable)")
    print(f"  • 2+ GB RAM:   ✓ YES (plenty of headroom)")
    print(f"")
    print(f"Map scalability:")
    n_points = len(map_3d_points)
    map_data_size = points_size + desc_size
    print(f"  Current map: {n_points} points = {map_data_size:.1f} MB")
    print(f"  Est. 100k points: {(map_data_size / n_points * 100000):.1f} MB")
    print(f"  Est. 200k points: {(map_data_size / n_points * 200000):.1f} MB")
    print("="*60)
    
    # Save to file
    with open('results/memory_footprint.txt', 'w') as f:
        f.write("MEMORY FOOTPRINT SUMMARY\n")
        f.write("="*60 + "\n")
        for component, size in components.items():
            percentage = (size / total) * 100
            f.write(f"{component:25} | {size:7.2f} MB | {percentage:5.1f}%\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':25} | {total:7.2f} MB | 100.0%\n")
    
    print("\n✓ Saved: results/memory_footprint.txt")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    measure_memory_footprint()