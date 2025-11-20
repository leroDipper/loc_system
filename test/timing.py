import numpy as np
import matplotlib.pyplot as plt
import os

def plot_timing_breakdown():
    """
    Generate timing breakdown figure (grouped bar chart).
    Shows feature extraction, matching, and PnP times on Raspberry Pi.
    Side-by-side bars make component comparison clear.
    """
    
    print("Loading timing data from Raspberry Pi experiments...")
    
    # Define which experiments to load
    experiments = {
        'FR1 FP32': 'results/fr1_fp32_errors.npz',
        'FR1 INT8': 'results/fr1_int8_errors.npz',
        'FR1 SIFT': 'results/fr1_sift_errors.npz',
        'FR3 FP32': 'results/fr3_fp32_errors.npz',
        'FR3 INT8': 'results/fr3_int8_errors.npz',
    }
    
    # Storage for timing data
    data = {}
    
    for name, filepath in experiments.items():
        if os.path.exists(filepath):
            loaded = np.load(filepath)
            
            # Check if timing data exists
            if 'timings_extract' in loaded.files:
                data[name] = {
                    'extract': loaded['timings_extract'] * 1000,  # Convert to ms
                    'match': loaded['timings_match'] * 1000,
                    'pnp': loaded['timings_pnp'] * 1000,
                }
                print(f"✓ Loaded {name}")
            else:
                print(f"✗ {filepath} missing timing data - re-run experiment with updated save code")
        else:
            print(f"✗ Missing {filepath}")
    
    if len(data) == 0:
        print("\n❌ No timing data found. Please re-run experiments with updated save code.")
        return
    
    # Compute statistics
    print("\n" + "="*60)
    print("TIMING STATISTICS (Raspberry Pi 5)")
    print("="*60)
    
    stats = {}
    for name, timings in data.items():
        extract_mean = np.mean(timings['extract'])
        match_mean = np.mean(timings['match'])
        pnp_mean = np.mean(timings['pnp'])
        total_mean = extract_mean + match_mean + pnp_mean
        fps = 1000.0 / total_mean
        
        stats[name] = {
            'extract': extract_mean,
            'match': match_mean,
            'pnp': pnp_mean,
            'total': total_mean,
            'fps': fps
        }
        
        print(f"\n{name}:")
        print(f"  Feature extraction: {extract_mean:6.2f} ms ({extract_mean/total_mean*100:.1f}%)")
        print(f"  Matching:           {match_mean:6.2f} ms ({match_mean/total_mean*100:.1f}%)")
        print(f"  PnP:                {pnp_mean:6.2f} ms ({pnp_mean/total_mean*100:.1f}%)")
        print(f"  Total:              {total_mean:6.2f} ms")
        print(f"  FPS:                {fps:6.2f}")
    
    # Compute speedups (if FP32 exists)
    if 'FR1 FP32' in stats and 'FR1 INT8' in stats:
        speedup = stats['FR1 FP32']['total'] / stats['FR1 INT8']['total']
        print(f"\nINT8 speedup over FP32 (FR1): {speedup:.2f}x")
    
    print("="*60)
    
    # ========== GROUPED BAR CHART ==========
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    # Focus on FR1 only for clarity (FR3 FP32 is too slow and compresses scale)
    methods = ['FP32', 'INT8', 'SIFT']
    components = ['Feature\nExtraction', 'Descriptor\nMatching', 'PnP\nRANSAC', 'Total\nPipeline']
    
    # Prepare data
    fp32_times = [stats['FR1 FP32']['extract'], stats['FR1 FP32']['match'], 
                  stats['FR1 FP32']['pnp'], stats['FR1 FP32']['total']]
    int8_times = [stats['FR1 INT8']['extract'], stats['FR1 INT8']['match'], 
                  stats['FR1 INT8']['pnp'], stats['FR1 INT8']['total']]
    sift_times = [stats['FR1 SIFT']['extract'], stats['FR1 SIFT']['match'], 
                  stats['FR1 SIFT']['pnp'], stats['FR1 SIFT']['total']]
    
    x = np.arange(len(components))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, fp32_times, width, label='XFeat-FP32', 
                   color='steelblue', edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x, int8_times, width, label='XFeat-INT8', 
                   color='orange', edgecolor='black', linewidth=1.2)
    bars3 = ax.bar(x + width, sift_times, width, label='SIFT', 
                   color='green', edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1, fp32_times)
    add_labels(bars2, int8_times)
    add_labels(bars3, sift_times)
    
    # Add FPS annotations for Total Pipeline
    total_idx = 3
    for i, (method, bar_offset) in enumerate(zip(['FP32', 'INT8', 'SIFT'], [-width, 0, width])):
        fps = stats[f'FR1 {method}']['fps']
        y_pos = stats[f'FR1 {method}']['total']
        ax.text(total_idx + bar_offset, y_pos + 10, f'{fps:.1f} FPS',
               ha='center', va='bottom', fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add 24 FPS threshold line (only on Total Pipeline section)
    ax.plot([2.5, 3.5], [42, 42], 'r--', linewidth=2.5, alpha=0.8, label='24 FPS threshold')
    ax.text(3.5, 43, '42ms', fontsize=9, color='red', ha='left', va='bottom',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add speedup annotations
    # Feature extraction speedup
    extract_speedup = fp32_times[0] / int8_times[0]
    ax.annotate(f'{extract_speedup:.1f}×', 
                xy=(0, max(fp32_times[0], int8_times[0])), 
                xytext=(0, max(fp32_times[0], int8_times[0]) + 15),
                ha='center', fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    
    # Total speedup
    total_speedup = fp32_times[3] / int8_times[3]
    ax.annotate(f'{total_speedup:.1f}× speedup', 
                xy=(3, max(fp32_times[3], int8_times[3])), 
                xytext=(3, max(fp32_times[3], int8_times[3]) + 15),
                ha='center', fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))
    
    # Formatting
    ax.set_ylabel('Time per Frame (ms)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Pipeline Component', fontsize=13, fontweight='bold')
    ax.set_title('Localization Pipeline Timing Breakdown on Raspberry Pi 5 (FR1 Dataset)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=11)
    ax.legend(fontsize=11, loc='center left', bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(max(fp32_times), max(int8_times), max(sift_times)) * 1.25)
    
    plt.tight_layout()
    plt.savefig('results/figure4_timing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure4_timing_breakdown.pdf', bbox_inches='tight')
    
    print("\n✓ Saved: results/figure4_timing_breakdown.png")
    print("✓ Saved: results/figure4_timing_breakdown.pdf")
    
    # Generate LaTeX table (FR1 only for main paper)
    print("\n" + "="*60)
    print("TABLE DATA FOR LATEX (FR1 only)")
    print("="*60)
    print("Method      | Extract | Match | PnP  | Total | FPS")
    print("------------|---------|-------|------|-------|-----")
    for method in ['FR1 FP32', 'FR1 INT8', 'FR1 SIFT']:
        if method not in stats:
            continue
        s = stats[method]
        method_short = method.replace('FR1 ', '')
        print(f"{method_short:11} | {s['extract']:6.2f}  | {s['match']:5.2f} | "
              f"{s['pnp']:4.2f} | {s['total']:5.2f} | {s['fps']:4.2f}")
    
    # Speedup analysis
    print("\n" + "="*60)
    print("SPEEDUP ANALYSIS (FP32 → INT8)")
    print("="*60)
    if 'FR1 FP32' in stats and 'FR1 INT8' in stats:
        fp32 = stats['FR1 FP32']
        int8 = stats['FR1 INT8']
        print(f"Component        | FP32    | INT8    | Speedup")
        print(f"-----------------|---------|---------|--------")
        print(f"Feature extract  | {fp32['extract']:6.2f}  | {int8['extract']:6.2f}  | {fp32['extract']/int8['extract']:.2f}x")
        print(f"Matching         | {fp32['match']:6.2f}  | {int8['match']:6.2f}  | {fp32['match']/int8['match']:.2f}x")
        print(f"PnP              | {fp32['pnp']:6.2f}  | {int8['pnp']:6.2f}  | {fp32['pnp']/int8['pnp']:.2f}x")
        print(f"TOTAL            | {fp32['total']:6.2f}  | {int8['total']:6.2f}  | {fp32['total']/int8['total']:.2f}x")
    
    # FR3 data (for supplementary/appendix)
    if 'FR3 FP32' in stats and 'FR3 INT8' in stats:
        print("\n" + "="*60)
        print("FR3 DATA (for appendix/text)")
        print("="*60)
        for method in ['FR3 FP32', 'FR3 INT8']:
            s = stats[method]
            print(f"{method}: {s['total']:.2f} ms ({s['fps']:.2f} FPS)")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_timing_breakdown()