import numpy as np
import matplotlib.pyplot as plt
import os

def plot_cross_precision():
    """
    Generate cross-precision matching comparison figure.
    Shows INT8 queries can match against FP32 maps with minimal degradation.
    """
    
    print("Loading cross-precision matching data...")
    
    # Define experiments (Query → Map configurations)
    experiments = {
        'FR1': {
            'FP32→FP32': 'results/fr1_fp32_errors.npz',
            'INT8→FP32': 'results/fr1_int8_errors.npz',
        },
        'FR3': {
            'FP32→FP32': 'results/fr3_fp32_errors.npz',
            'INT8→FP32': 'results/fr3_int8_errors.npz',
        }
    }
    
    # Load data
    data = {}
    for dataset, configs in experiments.items():
        data[dataset] = {}
        for config_name, filepath in configs.items():
            if os.path.exists(filepath):
                loaded = np.load(filepath)
                data[dataset][config_name] = {
                    'errors': loaded['errors'],
                    'success_rate': loaded['success_rate'] * 100,
                    'match_counts': loaded['match_counts'] if 'match_counts' in loaded.files else None
                }
                print(f"✓ Loaded {dataset} {config_name}")
            else:
                print(f"✗ Missing {filepath}")
    
    # Check if we have data
    if len(data['FR1']) == 0 and len(data['FR3']) == 0:
        print("\n❌ No data found. Please run experiments first.")
        return
    
    # Compute statistics
    print("\n" + "="*60)
    print("CROSS-PRECISION MATCHING STATISTICS")
    print("="*60)
    
    stats = {}
    for dataset in ['FR1', 'FR3']:
        if dataset not in data or len(data[dataset]) == 0:
            continue
        
        stats[dataset] = {}
        for config_name, config_data in data[dataset].items():
            errors = config_data['errors']
            
            median_cm = np.median(errors) * 100
            mean_cm = np.mean(errors) * 100
            below_20cm = np.sum(errors <= 0.20) / len(errors) * 100
            success_rate = config_data['success_rate']
            
            stats[dataset][config_name] = {
                'success_rate': success_rate,
                'median_cm': median_cm,
                'mean_cm': mean_cm,
                'below_20cm': below_20cm,
            }
            
            print(f"\n{dataset} {config_name}:")
            print(f"  Success rate:   {success_rate:.1f}%")
            print(f"  Median error:   {median_cm:.2f} cm")
            print(f"  Mean error:     {mean_cm:.2f} cm")
            print(f"  <20cm:          {below_20cm:.1f}%")
    
    # Compute degradation
    for dataset in ['FR1', 'FR3']:
        if dataset not in stats:
            continue
        if 'FP32→FP32' in stats[dataset] and 'INT8→FP32' in stats[dataset]:
            fp32 = stats[dataset]['FP32→FP32']
            int8 = stats[dataset]['INT8→FP32']
            
            success_deg = int8['success_rate'] - fp32['success_rate']
            median_deg = int8['median_cm'] - fp32['median_cm']
            below20_deg = int8['below_20cm'] - fp32['below_20cm']
            
            print(f"\n{dataset} Degradation (INT8 - FP32):")
            print(f"  Success rate:   {success_deg:+.1f} percentage points")
            print(f"  Median error:   {median_deg:+.2f} cm")
            print(f"  <20cm:          {below20_deg:+.1f} percentage points")
    
    print("="*60)
    
    # ========== CREATE FIGURE ==========
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ('success_rate', 'Success Rate (%)', 'higher_better'),
        ('median_cm', 'Median Error (cm)', 'lower_better'),
        ('below_20cm', 'Frames < 20cm (%)', 'higher_better')
    ]
    
    # Prepare data for plotting
    datasets_available = [d for d in ['FR1', 'FR3'] if d in stats and len(stats[d]) > 0]
    x = np.arange(len(datasets_available))
    width = 0.35
    
    for ax_idx, (metric_key, metric_label, direction) in enumerate(metrics):
        ax = axes[ax_idx]
        
        fp32_values = [stats[d]['FP32→FP32'][metric_key] for d in datasets_available]
        int8_values = [stats[d]['INT8→FP32'][metric_key] for d in datasets_available]
        
        bars1 = ax.bar(x - width/2, fp32_values, width, label='FP32→FP32', 
                      color='steelblue', edgecolor='black', linewidth=1.2)
        bars2 = ax.bar(x + width/2, int8_values, width, label='INT8→FP32', 
                      color='orange', edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + (0.5 if metric_key != 'median_cm' else 0.3),
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add degradation annotations
        for i, dataset in enumerate(datasets_available):
            fp32_val = stats[dataset]['FP32→FP32'][metric_key]
            int8_val = stats[dataset]['INT8→FP32'][metric_key]
            diff = int8_val - fp32_val
            
            # Position annotation between bars
            y_pos = max(fp32_val, int8_val) + (3 if metric_key != 'median_cm' else 0.5)
            
            color = 'green' if abs(diff) < 2 else 'orange'
           
        # Formatting
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(97+ax_idx)}) {metric_label}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(datasets_available, fontsize=11)
        ax.legend(fontsize=10, loc='best')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Set y-limits with some headroom
        if metric_key == 'median_cm':
            ax.set_ylim(0, max(max(fp32_values), max(int8_values)) * 1.3)
        else:
            ax.set_ylim(0, 105)
    
    plt.suptitle('Cross-Precision Matching: INT8 Queries vs FP32 Map', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plt.savefig('results/figure5_cross_precision.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure5_cross_precision.pdf', bbox_inches='tight')
    
    print("\n✓ Saved: results/figure5_cross_precision.png")
    print("✓ Saved: results/figure5_cross_precision.pdf")
    
    # Generate LaTeX table
    print("\n" + "="*60)
    print("TABLE DATA FOR LATEX")
    print("="*60)
    print("Dataset | Config      | Success% | Median (cm) | <20cm%")
    print("--------|-------------|----------|-------------|-------")
    
    for dataset in datasets_available:
        for config in ['FP32→FP32', 'INT8→FP32']:
            s = stats[dataset][config]
            print(f"{dataset:7} | {config:11} | {s['success_rate']:7.1f}% | "
                  f"{s['median_cm']:10.2f} | {s['below_20cm']:6.1f}%")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_cross_precision()