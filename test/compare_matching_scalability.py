"""
compare_matching_scalability.py

Compare brute-force vs vocabulary tree matching as map size increases.
Builds progressively larger maps and measures matching time.

Dataset: TUM FR3 Long Office
"""

import numpy as np
import time
import os
from loc_modules.map_builder import MapBuilder
from aux_tasks.vocabTree import VocabTreeBuilder
from accelerated_modules import vocab_tree_match
from .brute_force_matcher import BruteForceMatcher, OpenCVBruteForceMatcher
import matplotlib.pyplot as plt

def build_map_and_vocab(n_images):
    """
    Build map and vocabulary tree using first n_images.
    
    Returns:
        map_descriptors: numpy array of map descriptors
        vocab_tree_path: path to saved vocab tree
    """
    print(f"\n{'='*60}")
    print(f"Building map with {n_images} images...")
    print(f"{'='*60}")
    
    # Build map
    map_builder = MapBuilder()
    map_3d_points, map_descriptors = map_builder.build_map_database(
        map_files='resources/tum_fr3/project_files',
        dataset_path='resources/tum_fr3/images',
        descriptors_path='resources/tum_fr3/project_files/descriptors',
        save_to=None,  # Don't save
        max_images=n_images
    )
    
    print(f"Map built: {len(map_3d_points)} 3D points, {len(map_descriptors)} descriptors")
    
    # Build vocabulary tree
    print("Building vocabulary tree...")
    n_branches = 10
    depth = 4
    descriptor_dim = map_descriptors.shape[1]
    
    builder = VocabTreeBuilder(n_branches, depth, descriptor_dim)
    builder.build(map_descriptors)
    
    vocab_tree_path = f'results/temp_vocab_tree_{n_images}.bin'
    builder.save(vocab_tree_path)
    
    return map_descriptors, vocab_tree_path


def benchmark_matching(map_descriptors, vocab_tree_path, query_descriptors, n_trials=10):
    """
    Benchmark all three matching methods.
    
    Returns:
        dict with timing results
    """
    print(f"\nBenchmarking with {len(map_descriptors)} map descriptors, "
          f"{len(query_descriptors)} query descriptors...")
    
    # Build matchers
    print("  Building vocab tree matcher...")
    vocab_matcher = vocab_tree_match.VocabTreeMatcher(vocab_tree_path, map_descriptors)
    
    print("  Building OpenCV brute-force matcher...")
    opencv_matcher = OpenCVBruteForceMatcher(map_descriptors)
    
    # Benchmark vocab tree
    times_vocab = []
    for _ in range(n_trials):
        t_start = time.time()
        query_idx, map_idx, distances = vocab_matcher.match(query_descriptors, ratio_threshold=0.80)
        t_elapsed = time.time() - t_start
        times_vocab.append(t_elapsed)
    
    vocab_time = np.mean(times_vocab) * 1000  # ms
    vocab_matches = len(query_idx)
    
    # Benchmark OpenCV brute-force (C++)
    times_opencv = []
    for _ in range(n_trials):
        t_start = time.time()
        query_idx, map_idx, distances = opencv_matcher.match(query_descriptors, ratio_threshold=0.80)
        t_elapsed = time.time() - t_start
        times_opencv.append(t_elapsed)
    
    opencv_time = np.mean(times_opencv) * 1000  # ms
    opencv_matches = len(query_idx)
    
    speedup = opencv_time / vocab_time
    
    print(f"  Vocab tree:   {vocab_time:.2f} ms ({vocab_matches} matches)")
    print(f"  OpenCV BF:    {opencv_time:.2f} ms ({opencv_matches} matches)")
    print(f"  Speedup:      {speedup:.2f}x")
    
    return {
        'vocab_time_ms': vocab_time,
        'opencv_time_ms': opencv_time,
        'speedup': speedup,
        'vocab_matches': vocab_matches,
        'opencv_matches': opencv_matches
    }


if __name__ == "__main__":
    
    # Map sizes to test (number of images to use for map building)
    # FR3 has ~2760 images total
    # Use increments that give good coverage
    map_sizes = [100, 250, 500, 750, 1000, 1500]
    
    # Load a fixed set of query descriptors (from test images)
    # Use images NOT in the map (e.g., last 100 images)
    print("Loading query descriptors from test images...")
    query_desc_path = 'resources/tum_fr3/project_files/descriptors'
    
    # Load one test image's descriptors
    test_image_name = '1341847980.722988.png'  # Pick any test image
    desc_file = os.path.join(query_desc_path, f"{test_image_name}_desc.txt")
    
    query_descriptors = []
    with open(desc_file, 'r') as f:
        for line in f:
            if line.strip():
                descriptor_array = np.array([int(x) for x in line.split()])
                query_descriptors.append(descriptor_array)
    
    query_descriptors = np.array(query_descriptors, dtype=np.float32)
    print(f"Loaded {len(query_descriptors)} query descriptors")
    
    # Storage for results
    results = {
        'map_sizes': [],
        'num_descriptors': [],
        'vocab_times': [],
        'opencv_times': [],
        'speedups': []
    }
    
    # Run experiments
    for n_images in map_sizes:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {n_images} images")
        print(f"{'='*60}")
        
        # Build map and vocab
        map_descriptors, vocab_tree_path = build_map_and_vocab(n_images)
        
        # Benchmark
        benchmark_results = benchmark_matching(
            map_descriptors, 
            vocab_tree_path, 
            query_descriptors,
            n_trials=5
        )
        
        # Store results
        results['map_sizes'].append(n_images)
        results['num_descriptors'].append(len(map_descriptors))
        results['vocab_times'].append(benchmark_results['vocab_time_ms'])
        results['opencv_times'].append(benchmark_results['opencv_time_ms'])
        results['speedups'].append(benchmark_results['speedup'])
        
        # Clean up temp vocab tree
        if os.path.exists(vocab_tree_path):
            os.remove(vocab_tree_path)
    
    # Save results
    os.makedirs('results', exist_ok=True)
    np.savez('results/matching_scalability.npz',
             map_sizes=np.array(results['map_sizes']),
             num_descriptors=np.array(results['num_descriptors']),
             vocab_times=np.array(results['vocab_times']),
             opencv_times=np.array(results['opencv_times']),
             speedups=np.array(results['speedups']))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Images':>7} | {'Descriptors':>12} | {'Vocab(ms)':>10} | {'OpenCV(ms)':>11} | {'Speedup':>8}")
    print("-"*65)
    for i in range(len(results['map_sizes'])):
        print(f"{results['map_sizes'][i]:7d} | {results['num_descriptors'][i]:12d} | "
              f"{results['vocab_times'][i]:10.2f} | {results['opencv_times'][i]:11.2f} | "
              f"{results['speedups'][i]:8.2f}x")
    
    print("\n✓ Results saved to results/matching_scalability.npz")
    
    # Generate simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Matching time vs map size
    ax1.plot(results['num_descriptors'], results['vocab_times'], 
             'o-', linewidth=2.5, markersize=10, label='Vocab Tree', color='steelblue')
    ax1.plot(results['num_descriptors'], results['opencv_times'], 
             's-', linewidth=2.5, markersize=10, label='OpenCV BF', color='orange')
    ax1.set_xlabel('Number of Map Descriptors', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Matching Time (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Matching Time vs Map Size', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Speedup vs map size
    ax2.plot(results['num_descriptors'], results['speedups'], 
             'D-', linewidth=2.5, markersize=10, color='green')
    ax2.set_xlabel('Number of Map Descriptors', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
    ax2.set_title('Vocab Tree Speedup over Brute-force', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/matching_scalability.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/matching_scalability.pdf', bbox_inches='tight')
    
    print("✓ Saved: results/matching_scalability.png")
    print("✓ Saved: results/matching_scalability.pdf")