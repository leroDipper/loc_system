"""
plot_resolution_variation.py

Simple plot: Speed vs Accuracy trade-off for different resolutions.
Two lines that cross showing the trade-off.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_resolution_variation():
    """
    Generate simple dual-axis plot:
    - FPS (decreasing with resolution)
    - Accuracy % (increasing with resolution)
    """
    
    print("Loading resolution variation data...")
    
    resolutions = [
        (160, 120),
        (320, 240),
        (480, 360),
        (640, 480),
        (800, 600),
        (960, 720),
    ]
    
    # Load data
    data = {}
    for res_w, res_h in resolutions:
        filepath = f'results/fr1_fp32_res_{res_w}x{res_h}_errors.npz'
        if os.path.exists(filepath):
            loaded = np.load(filepath)
            
            errors = loaded['errors']
            timings_extract = loaded['timings_extract']
            timings_match = loaded['timings_match']
            timings_pnp = loaded['timings_pnp']
            
            # Compute statistics
            median_error_cm = np.median(errors) * 100
            mean_error_cm = np.mean(errors) * 100
            success_rate = loaded['success_rate'] * 100
            
            mean_extract_ms = np.mean(timings_extract) * 1000
            mean_match_ms = np.mean(timings_match) * 1000
            mean_pnp_ms = np.mean(timings_pnp) * 1000
            mean_total_ms = mean_extract_ms + mean_match_ms + mean_pnp_ms
            fps = 1000.0 / mean_total_ms
            
            data[f"{res_w}x{res_h}"] = {
                'median_error_cm': median_error_cm,
                'mean_error_cm': mean_error_cm,
                'success_rate': success_rate,
                'fps': fps,
                'extract_ms': mean_extract_ms,
                'match_ms': mean_match_ms,
                'pnp_ms': mean_pnp_ms,
                'total_ms': mean_total_ms,
                'pixels': res_w * res_h
            }
            
            print(f"✓ Loaded {res_w}x{res_h}: {median_error_cm:.2f}cm @ {fps:.2f} FPS")
        else:
            print(f"✗ Missing {filepath}")
    
    if len(data) == 0:
        print("\n❌ No data found. Please run vary_resolution_fr1.py first.")
        return
    
    # Extract arrays for plotting
    res_labels = [f"{w}x{h}" for w, h in resolutions if f"{w}x{h}" in data]
    pixels = [data[r]['pixels'] for r in res_labels]
    median_errors = [data[r]['median_error_cm'] for r in res_labels]
    fpss = [data[r]['fps'] for r in res_labels]
    
    # Convert median error to accuracy percentage (inverse relationship)
    # Accuracy % = 100 - (error/max_error)*100
    max_error = 20.0  # cm
    accuracy_pct = [100 - (err / max_error) * 100 for err in median_errors]
    
    # Print summary
    print("\n" + "="*60)
    print("RESOLUTION VARIATION SUMMARY")
    print("="*60)
    print(f"{'Resolution':>12} | {'Error(cm)':>10} | {'Accuracy(%)':>12} | {'FPS':>7}")
    print("-"*60)
    for r, err, acc in zip(res_labels, median_errors, accuracy_pct):
        d = data[r]
        print(f"{r:>12} | {err:10.2f} | {acc:12.1f} | {d['fps']:7.2f}")
    print("="*60)
    
    # ========== CREATE SIMPLE DUAL-AXIS PLOT ==========
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot FPS (left y-axis, decreasing)
    color1 = 'tab:blue'
    ax1.set_xlabel('Image Resolution (pixels)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frame Rate (FPS)', fontsize=13, fontweight='bold', color=color1)
    ax1.plot(pixels, fpss, 'o-', linewidth=3, markersize=12, 
             color=color1, label='Speed (FPS)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis with resolution labels
    ax1.set_xticks(pixels)
    ax1.set_xticklabels(res_labels, rotation=45, ha='right', fontsize=10)
    
    # Plot Accuracy percentage (right y-axis, increasing)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Localization Accuracy (%)', fontsize=13, fontweight='bold', color=color2)
    ax2.plot(pixels, accuracy_pct, 's-', linewidth=3, markersize=12,
             color=color2, label='Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax1.set_title('Resolution Trade-off: Speed vs Accuracy', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/figure4_resolution_variation.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure4_resolution_variation.pdf', bbox_inches='tight')
    
    print("\n✓ Saved: results/figure4_resolution_variation.png")
    print("✓ Saved: results/figure4_resolution_variation.pdf")
    
    # ========== GENERATE LATEX TABLE ==========
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Impact of resolution on localization performance (TUM FR1, XFeat CPU, N=200)}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("\\textbf{Resolution} & \\textbf{Error (cm)} & \\textbf{Accuracy (\\%)} & \\textbf{FPS} & \\textbf{Total (ms)} \\\\")
    print("\\midrule")
    
    for r in res_labels:
        d = data[r]
        acc = 100 - (d['median_error_cm'] / 20.0) * 100
        print(f"{r:12} & {d['median_error_cm']:6.2f} & {acc:6.1f} & "
              f"{d['fps']:5.2f} & {d['total_ms']:6.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\label{tab:resolution_variation}")
    print("\\end{table}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_resolution_variation()
