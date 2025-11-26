import numpy as np

data = np.load('resources/tum_fr1/map_databases/tumfr1_map_train_pruned50.npz')
desc = data['descriptors']

print(f"Shape: {desc.shape}")
print(f"Unique descriptors: {len(np.unique(desc, axis=0))}")
print(f"Descriptor sample:\n{desc[:5]}")