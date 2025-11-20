import numpy as np
import matplotlib.pyplot as plt
import os

def plot_timing_breakdown():
    """
    Generate timing breakdown figure (stacked bar chart).
    Shows feature extraction, matching, and PnP times on Raspberry Pi.
    """
    
    print("Loading timing data from Raspberry Pi experiments...")
    
    # Define which experiments to load
    experiments = {
        'FR1 INT8': 'results/fr1_int8_errors.npz',
        'FR1 SIFT': 'results/fr1_sift_errors.npz',
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
    
    print("="*60)
    
    # Create stacked bar chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    methods = list(stats.keys())
    x_pos = np.arange(len(methods))
    
    extract_times = [stats[m]['extract'] for m in methods]
    match_times = [stats[m]['match'] for m in methods]
    pnp_times = [stats[m]['pnp'] for m in methods]
    
    # Stacked bars
    bar_width = 0.6
    p1 = ax.bar(x_pos, extract_times, bar_width, label='Feature Extraction', 
                color='steelblue', edgecolor='black')
    p2 = ax.bar(x_pos, match_times, bar_width, bottom=extract_times, 
                label='Descriptor Matching', color='orange', edgecolor='black')
    p3 = ax.bar(x_pos, pnp_times, bar_width, 
                bottom=np.array(extract_times) + np.array(match_times),
                label='PnP RANSAC', color='lightgreen', edgecolor='black')
    
    # Add total time and FPS annotations on top of bars
    for i, method in enumerate(methods):
        total = stats[method]['total']
        fps = stats[method]['fps']
        ax.text(i, total + 2, f'{total:.1f} ms\n{fps:.1f} FPS', 
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add horizontal line at 42ms (24 FPS threshold)
    ax.axhline(y=42, color='red', linestyle='--', linewidth=2, alpha=0.7, 
               label='24 FPS threshold (42ms)')
    
    # Formatting
    ax.set_ylabel('Time per Frame (ms)', fontsize=13)
    ax.set_xlabel('Method', fontsize=13)
    ax.set_title('Localization Pipeline Timing Breakdown (Raspberry Pi 5)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max([stats[m]['total'] for m in methods]) * 1.15)
    
    plt.tight_layout()
    plt.savefig('results/figure4_timing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure4_timing_breakdown.pdf', bbox_inches='tight')
    
    print("\n✓ Saved: results/figure4_timing_breakdown.png")
    print("✓ Saved: results/figure4_timing_breakdown.pdf")
    
    # Generate LaTeX table
    print("\n" + "="*60)
    print("TABLE DATA FOR LATEX")
    print("="*60)
    print("Method      | Extract | Match | PnP  | Total | FPS")
    print("------------|---------|-------|------|-------|-----")
    for method in methods:
        s = stats[method]
        print(f"{method:11} | {s['extract']:6.2f}  | {s['match']:5.2f} | "
              f"{s['pnp']:4.2f} | {s['total']:5.2f} | {s['fps']:4.2f}")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_timing_breakdown()