import numpy as np
import matplotlib.pyplot as plt
import os

def plot_error_cdf():
    """
    Generate Error CDF plot for journal Figure 2.
    Compares FP32, INT8, and SIFT across FR1 and FR3 datasets.
    """
    
    # Load error data
    print("Loading error data...")
    
    datasets = {
        'FR1': {
            'FP32': 'results/fr1_fp32_errors.npz',
            'INT8': 'results/fr1_int8_errors.npz',
            'SIFT': 'results/fr1_sift_errors.npz',
        },
        'FR3': {
            'FP32': 'results/fr3_fp32_errors.npz',
            'INT8': 'results/fr3_int8_errors.npz',
        }
    }
    
    # Check which files exist
    data = {}
    for dataset_name, methods in datasets.items():
        data[dataset_name] = {}
        for method_name, filepath in methods.items():
            if os.path.exists(filepath):
                loaded = np.load(filepath)
                data[dataset_name][method_name] = loaded['errors']
                print(f"✓ Loaded {dataset_name} {method_name}: {len(loaded['errors'])} errors")
            else:
                print(f"✗ Missing {filepath}")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    colors = {'FP32': 'blue', 'INT8': 'orange', 'SIFT': 'green'}
    linestyles = {'FR1': '-', 'FR3': '--'}
    
    # Plot CDFs
    for dataset_name in ['FR1', 'FR3']:
        if dataset_name not in data:
            continue
            
        for method_name in ['FP32', 'INT8', 'SIFT']:
            if method_name not in data[dataset_name]:
                continue
            
            errors = data[dataset_name][method_name]
            
            # Sort errors
            sorted_errors = np.sort(errors)
            
            # Compute CDF (percentage of frames)
            cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
            
            # Plot
            label = f"{method_name} ({dataset_name})"
            ax.plot(sorted_errors, cdf, 
                   color=colors[method_name],
                   linestyle=linestyles[dataset_name],
                   linewidth=2.5,
                   label=label)
            
            # Print key statistics
            print(f"\n{label}:")
            print(f"  Median: {np.median(errors):.4f} m ({np.median(errors)*100:.2f} cm)")
            print(f"  Mean:   {np.mean(errors):.4f} m ({np.mean(errors)*100:.2f} cm)")
            print(f"  @ 5cm:  {np.sum(errors <= 0.05)/len(errors)*100:.1f}%")
            print(f"  @ 10cm: {np.sum(errors <= 0.10)/len(errors)*100:.1f}%")
            print(f"  @ 20cm: {np.sum(errors <= 0.20)/len(errors)*100:.1f}%")
    
    # Add threshold lines
    thresholds = [0.05, 0.10, 0.20]
    threshold_labels = ['5cm', '10cm', '20cm']
    for thresh, label in zip(thresholds, threshold_labels):
        ax.axvline(thresh, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
        ax.text(thresh, 5, label, rotation=0, fontsize=10, 
               verticalalignment='bottom', horizontalalignment='center',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel('Localisation Error (m)', fontsize=13)
    ax.set_ylabel('Cumulative Percentage of Frames (%)', fontsize=13)
    ax.set_title('Localization Error CDF: XFeat vs SIFT', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11, loc='lower right')
    ax.set_xlim(0, 0.5)  # Focus on 0-50cm range
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('results/figure2_error_cdf.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/figure2_error_cdf.pdf', bbox_inches='tight')
    
    print("\n" + "="*60)
    print("✓ Saved: results/figure2_error_cdf.png")
    print("✓ Saved: results/figure2_error_cdf.pdf")
    print("="*60)
    
    # Generate summary table
    print("\n" + "="*60)
    print("TABLE DATA FOR LATEX")
    print("="*60)
    print("Dataset | Method | Success% | Median (cm) | Mean (cm) | <20cm%")
    print("--------|--------|----------|-------------|-----------|-------")
    
    for dataset_name in ['FR1', 'FR3']:
        if dataset_name not in data:
            continue
        for method_name in ['FP32', 'INT8', 'SIFT']:
            if method_name not in data[dataset_name]:
                continue
            
            errors = data[dataset_name][method_name]
            # Load success rate
            filepath = datasets[dataset_name][method_name]
            loaded = np.load(filepath)
            success_rate = loaded['success_rate'] * 100
            
            median_cm = np.median(errors) * 100
            mean_cm = np.mean(errors) * 100
            below_20cm = np.sum(errors <= 0.20) / len(errors) * 100
            
            print(f"{dataset_name:7} | {method_name:6} | {success_rate:7.1f}% | "
                  f"{median_cm:10.2f} | {mean_cm:9.2f} | {below_20cm:6.1f}%")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    plot_error_cdf()