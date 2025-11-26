"""
compare_matching_accuracy.py

Simple comparison of vocab tree vs brute-force matching accuracy.
Measures: match count, overlap, and localization impact.
"""

import numpy as np
import os
from accelerated_modules import vocab_tree_match
from test.brute_force_matcher import OpenCVBruteForceMatcher
import glob

if __name__ == "__main__":
    
    print("="*60)
    print("VOCAB TREE vs BRUTE-FORCE ACCURACY COMPARISON")
    print("="*60)
    
    # Load map
    print("\nLoading map...")
    data = np.load('resources/tum_fr1/map_databases/tumfr1_map_train.npz')
    map_descriptors = data['descriptors']
    print(f"Map: {len(map_descriptors)} descriptors")
    
    # Load vocabulary tree
    vocabulary = 'resources/tum_fr1/vocabularies/vocab_tree.bin'
    vocab_matcher = vocab_tree_match.VocabTreeMatcher(vocabulary, map_descriptors)
    
    # Build brute-force matcher
    brute_matcher = OpenCVBruteForceMatcher(map_descriptors)
    
    # Load test query descriptors (pick 10 random test images)
    print("\nLoading query images...")
    test_dataset_path = 'resources/tum_fr1/images'
    all_frames = sorted(glob.glob(os.path.join(test_dataset_path, "*.png")))
    test_frames = all_frames[500:510]  # 10 test images
    
    descriptors_path = 'resources/tum_fr1/project_files/descriptors'
    
    total_vocab_matches = 0
    total_brute_matches = 0
    total_overlap = 0
    
    print(f"Testing on {len(test_frames)} images...\n")
    
    for frame_path in test_frames:
        frame_name = os.path.basename(frame_path)
        desc_file = os.path.join(descriptors_path, f"{frame_name}_desc.txt")
        
        # Load query descriptors
        query_descriptors = []
        with open(desc_file, 'r') as f:
            for line in f:
                if line.strip():
                    descriptor_array = np.array([int(x) for x in line.split()])
                    query_descriptors.append(descriptor_array)
        
        query_descriptors = np.array(query_descriptors[:200], dtype=np.float32)  # Use first 200
        
        # Match with vocab tree
        vocab_query_idx, vocab_map_idx, vocab_dist = vocab_matcher.match(
            query_descriptors, ratio_threshold=0.80
        )
        
        # Match with brute-force
        brute_query_idx, brute_map_idx, brute_dist = brute_matcher.match(
            query_descriptors, ratio_threshold=0.80
        )
        
        # Compute overlap
        vocab_pairs = set(zip(vocab_query_idx, vocab_map_idx))
        brute_pairs = set(zip(brute_query_idx, brute_map_idx))
        overlap = len(vocab_pairs & brute_pairs)
        
        total_vocab_matches += len(vocab_query_idx)
        total_brute_matches += len(brute_query_idx)
        total_overlap += overlap
        
        print(f"{frame_name}: Vocab={len(vocab_query_idx)}, Brute={len(brute_query_idx)}, Overlap={overlap}")
    
    # Summary
    avg_vocab = total_vocab_matches / len(test_frames)
    avg_brute = total_brute_matches / len(test_frames)
    overlap_pct = (total_overlap / total_brute_matches) * 100
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Average matches per image:")
    print(f"  Vocab Tree:   {avg_vocab:.1f}")
    print(f"  Brute-force:  {avg_brute:.1f}")
    print(f"  Difference:   {avg_vocab - avg_brute:+.1f} ({(avg_vocab/avg_brute - 1)*100:+.1f}%)")
    print(f"\nMatch overlap:  {total_overlap}/{total_brute_matches} ({overlap_pct:.1f}%)")
    print(f"\nConclusion: Vocab tree recovers {overlap_pct:.1f}% of brute-force matches")
    print("="*60)
    
    # Save results
    np.savez('results/matching_accuracy_comparison.npz',
             avg_vocab_matches=avg_vocab,
             avg_brute_matches=avg_brute,
             overlap_percentage=overlap_pct)
    
    print("\n✓ Saved: results/matching_accuracy_comparison.npz")