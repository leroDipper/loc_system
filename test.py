# test_vocab_tree.py
import numpy as np
from accelerated_modules import vocab_tree_match

print("Loading map data...")
data = np.load('resources/map_databases/colmap_map_train_set.npz')
map_descriptors = data['descriptors'].astype(np.float32)
print(f"Loaded {len(map_descriptors)} map descriptors")

print("\nCreating VocabTreeMatcher...")
matcher = vocab_tree_match.VocabTreeMatcher(
    'resources/vocabularies/vocab_tree.bin',
    map_descriptors
)

print("\nTesting with a few query descriptors...")
# Use some map descriptors as queries for testing
query_descriptors = map_descriptors[:100].copy()

query_idx, map_idx, distances = matcher.match(query_descriptors, ratio_threshold=0.80)

print(f"\nResults:")
print(f"  Query descriptors: {len(query_descriptors)}")
print(f"  Matches found: {len(query_idx)}")
print(f"  Match rate: {len(query_idx)/len(query_descriptors)*100:.1f}%")

if len(query_idx) > 0:
    print(f"\nFirst 5 matches:")
    for i in range(min(5, len(query_idx))):
        print(f"  Query {query_idx[i]} → Map {map_idx[i]}, dist={distances[i]:.4f}")