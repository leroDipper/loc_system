"""
plot_feature_variation.py

Visualize accuracy-speed trade-off from feature count variation experiment.
Creates figure for Section 3 (methodology/design exploration).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_feature_variation():
    """
    Generate 3-panel figure showing:
    (a) Accuracy vs N
    (b) Speed vs N  
    (c) Pareto frontier (accuracy-speed trade-off)
    """
    
    print("Loading feature variation data...")
    
    feature_counts = [50, 100, 200, 500, 1000, 1500, 2000]
    
    # Load data
    data = {}
    for N in feature_counts:
        filepath = f'results/fr1_fp32_N{N}_errors.npz'
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
            
            data[N] = {
                'median_error_cm': median_error_cm,
                'mean_error_cm': mean_error_cm,
                'success_rate': success_rate,
                'fps': fps,
                'extract_ms': mean_extract_ms,
                'match_ms': mean_match_ms,
                'pnp_ms': mean_pnp_ms,
                'total_ms': mean_total_ms
            }
            
            print(f"✓ Loaded N={N}: {median_error_cm:.2f}cm @ {fps:.2f} FPS")
        else:
            print(f"✗ Missing {filepath}")
    
    if len(data) == 0:
        print("\n❌ No data found. Please run vary_features_fr1.py first.")
        return
    
    # Extract arrays for plotting
    Ns = sorted(data.keys())
    median_errors = [data[N]['median_error_cm'] for N in Ns]
    fpss = [data[N]['fps'] for N in Ns]
    
    # Convert median error to accuracy percentage (inverse relationship)
    # Accuracy % = 100 - (error/max_error)*100
    # So lower error = higher accuracy percentage
    max_error = 20.0  # cm
    accuracy_pct = [100 - (err / max_error) * 100 for err in median_errors]
    
    # Print summary
    print("\n" + "="*60)
    print("FEATURE VARIATION SUMMARY")
    print("="*60)
    print(f"{'N':>6} | {'Error(cm)':>10} | {'Accuracy(%)':>12} | {'FPS':>7}")
    print("-"*60)
    for N, err, acc in zip(Ns, median_errors, accuracy_pct):
        d = data[N]
        print(f"{N:6d} | {err:10.2f} | {acc:12.1f} | {d['fps']:7.2f}")
    print("="*60)
    
    # ========== CREATE SIMPLE DUAL-AXIS PLOT ==========
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot FPS (left y-axis, decreasing)
    color1 = 'tab:blue'
    ax1.set_xlabel('Number of Features (N)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frame Rate (FPS)', fontsize=13, fontweight='bold', color=color1)
    ax1.plot(Ns, fpss, 'o-', linewidth=3, markersize=12, 
             color=color1, label='Speed (FPS)')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xscale('log')
    ax1.set_xticks(Ns)
    ax1.set_xticklabels(Ns, fontsize=11)
    
    # Plot Accuracy percentage (right y-axis, increasing)
    ax2 = ax1.twinx()
    color2 = 'tab:orange'
    ax2.set_ylabel('Localization Accuracy (%)', fontsize=13, fontweight='bold', color=color2)
    ax2.plot(Ns, accuracy_pct, 's-', linewidth=3, markersize=12,
             color=color2, label='Accuracy (%)')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax1.set_title('Feature Count Trade-off: Speed vs Accuracy', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    # Save
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/figure3_feature_variation.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure3_feature_variation.pdf', bbox_inches='tight')
    
    print("\n✓ Saved: results/figure3_feature_variation.png")
    print("✓ Saved: results/figure3_feature_variation.pdf")
    
    # ========== GENERATE LATEX TABLE ==========
    print("\n" + "="*60)
    print("LATEX TABLE")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Impact of feature count on localization performance (TUM FR1, XFeat CPU)}")
    print("\\begin{tabular}{rrrrrr}")
    print("\\toprule")
    print("\\textbf{N} & \\textbf{Error (cm)} & \\textbf{FPS} & \\textbf{Extract (ms)} & \\textbf{Match (ms)} & \\textbf{PnP (ms)} \\\\")
    print("\\midrule")
    
    for N in Ns:
        d = data[N]
        print(f"{N:4d} & {d['median_error_cm']:6.2f} & {d['fps']:5.2f} & "
              f"{d['extract_ms']:6.2f} & {d['match_ms']:5.2f} & {d['pnp_ms']:5.2f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\label{tab:feature_variation}")
    print("\\end{table}")


if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_feature_variation()