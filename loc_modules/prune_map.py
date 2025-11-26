"""
Müller's coverage-based map pruning using sparse matrix optimization.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csr_matrix

class MapPruner:
    """Prune map using Müller's coverage-based greedy selection."""
    
    def __init__(self, map_npz_path, colmap_path, b_cover=50):
        """
        Args:
            map_npz_path: Path to .npz file with xyz_world and descriptors
            colmap_path: Path to COLMAP project directory (contains points3D.txt, images.txt)
            b_cover: Coverage target (min points per frame)
        """
        self.b_cover = b_cover
        
        # Load map
        print(f"Loading map from {map_npz_path}")
        data = np.load(map_npz_path)
        self.xyz = data['xyz_world']
        self.descriptors = data['descriptors']
        self.n_points = len(self.xyz)
        print(f"  Loaded {self.n_points} points")
        print(f"  Descriptor shape: {self.descriptors.shape}")
        
        # Verify descriptors are unique
        unique_count = len(np.unique(self.descriptors, axis=0))
        print(f"  Unique descriptors in original map: {unique_count}/{self.n_points}")
        
        # Build point ID mapping
        print(f"\nRebuilding point mapping from COLMAP data...")
        self.co_visibility = self._build_point_mapping(colmap_path)
        print(f"  Mapped {len(self.co_visibility)} points")
        
        # Get unique training images
        all_images = set()
        for img_ids in self.co_visibility.values():
            all_images.update(img_ids)
        self.training_images = sorted(all_images)
        print(f"  Found {len(self.training_images)} training images")
    
    def _build_point_mapping(self, colmap_path):
        """Build point mapping by replicating map_builder.py logic."""
        colmap_path = Path(colmap_path)
        
        # Load points3D.txt
        points = []
        with open(colmap_path / "points3D.txt", 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 8:
                    point_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    
                    # Parse track
                    track = []
                    for i in range(8, len(parts), 2):
                        if i + 1 < len(parts):
                            img_id = int(parts[i])
                            kp_idx = int(parts[i+1])
                            track.append((img_id, kp_idx))
                    
                    points.append({
                        'id': point_id,
                        'xyz': np.array([x, y, z]),
                        'track': track
                    })
        
        # Load images.txt
        images_data = {}
        with open(colmap_path / "images.txt", 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split()
                if len(parts) == 10:
                    img_id = int(parts[0])
                    img_name = parts[9]
                    images_data[img_id] = img_name
        
        # Build co-visibility in the same order as map_builder.py
        co_visibility = {}
        map_idx = 0
        
        for point in points:
            if not point['track']:
                continue
            
            first_img_id, first_kp_idx = point['track'][0]
            img_name = images_data.get(first_img_id)
            
            if not img_name:
                continue
            
            # Safety check
            if map_idx >= self.n_points:
                break
            
            # Extract image IDs from track
            track_img_ids = [img_id for img_id, kp_idx in point['track']]
            
            # Store mapping
            co_visibility[map_idx] = track_img_ids
            map_idx += 1
        
        return co_visibility
    
    def build_visibility_matrix(self):
        """Build sparse point × image visibility matrix (CSR format)."""
        print("Building sparse visibility matrix...")
        
        num_points = self.n_points
        num_images = len(self.training_images)
        img_to_idx = {img_id: i for i, img_id in enumerate(self.training_images)}
        
        rows = []
        cols = []
        
        for p_idx, img_ids in self.co_visibility.items():
            for img_id in img_ids:
                if img_id in img_to_idx:
                    rows.append(p_idx)
                    cols.append(img_to_idx[img_id])
        
        rows = np.array(rows)
        cols = np.array(cols)
        data = np.ones(len(rows), dtype=np.uint8)
        
        V = csr_matrix((data, (rows, cols)),
                       shape=(num_points, num_images),
                       dtype=np.uint8)
        
        print(f"  Sparse matrix built: {V.shape}, nnz={V.nnz}")
        return V
    
    def greedy_select_sparse(self, k):
        """Optimized greedy selection using sparse CSR visibility matrix."""
        
        # Build visibility matrix
        V = self.build_visibility_matrix()
        num_points, num_images = V.shape
        
        # Track frames' current coverage counts
        current_cov = np.zeros(num_images, dtype=np.int32)
        selected = []
        remaining = np.ones(num_points, dtype=bool)
        
        print(f"\n{'='*60}")
        print(f"SPARSE GREEDY SELECTION")
        print(f"{'='*60}")
        print(f"Selecting {k}/{num_points} points")
        print(f"Coverage target: {self.b_cover} points per frame")
        
        for i in tqdm(range(k), desc="Selecting points"):
            # Compute which image slots still give extra coverage
            remaining_capacity = (current_cov < self.b_cover).astype(np.uint8)
            
            # Compute marginal gain for REMAINING points only
            # Get indices of remaining points
            remaining_indices = np.where(remaining)[0]
            
            if len(remaining_indices) == 0:
                print(f"\nStopping early at {len(selected)} points (no remaining points)")
                break
            
            # Compute gains only for remaining points
            V_remaining = V[remaining_indices, :]
            gains_remaining = (V_remaining.multiply(remaining_capacity)).sum(axis=1).A.squeeze()
            
            # Handle 1D case
            if gains_remaining.ndim == 0:
                gains_remaining = np.array([gains_remaining])
            
            # Find best among remaining
            best_relative_idx = gains_remaining.argmax()
            best_gain = gains_remaining[best_relative_idx]
            
            if best_gain <= 0:
                print(f"\nStopping early at {len(selected)} points (no more gainful points)")
                break
            
            # Map back to original index
            best_idx = remaining_indices[best_relative_idx]
            
            selected.append(int(best_idx))
            remaining[best_idx] = False
            
            # Update coverage using sparse row (fast)
            img_indices = V[best_idx].indices  # images where point is visible
            current_cov[img_indices] += 1
            
            # Progress update
            if (i + 1) % 5000 == 0:
                print(f"\n  Progress: {i+1}/{k} points")
                print(f"    Avg coverage: {current_cov.mean():.1f}")
                print(f"    Min coverage: {current_cov.min()}")
                print(f"    Max coverage: {current_cov.max()}")
                print(f"    Remaining points: {remaining.sum()}")
        
        print(f"\n{'='*60}")
        print(f"SELECTION COMPLETE")
        print(f"{'='*60}")
        print(f"Selected: {len(selected)} points")
        print(f"Coverage stats:")
        print(f"  Average: {current_cov.mean():.1f} points per frame")
        print(f"  Min:     {current_cov.min()} points per frame")
        print(f"  Max:     {current_cov.max()} points per frame")
        
        selected_array = np.array(selected, dtype=np.int32)
        
        # Verify uniqueness
        unique_selected = len(np.unique(selected_array))
        print(f"\nVerification:")
        print(f"  Total selected: {len(selected_array)}")
        print(f"  Unique selected: {unique_selected}")
        
        if unique_selected < len(selected_array):
            print(f"  ERROR: Duplicate selections detected!")
        
        return selected_array
    
    def save_pruned_map(self, selected_indices, output_path):
        """Save pruned map with selected points."""
        
        print(f"\n{'='*60}")
        print(f"SAVING PRUNED MAP")
        print(f"{'='*60}")
        
        # Verify indices
        print(f"Selected indices shape: {selected_indices.shape}")
        print(f"Min index: {selected_indices.min()}")
        print(f"Max index: {selected_indices.max()}")
        print(f"Unique indices: {len(np.unique(selected_indices))}")
        print(f"First 10 indices: {selected_indices[:10]}")
        
        # Index into arrays
        selected_xyz = self.xyz[selected_indices]
        selected_desc = self.descriptors[selected_indices]
        
        # Verify descriptors are different
        unique_desc = len(np.unique(selected_desc, axis=0))
        print(f"Unique descriptors in selection: {unique_desc}/{len(selected_indices)}")
        
        if unique_desc < len(selected_indices) * 0.9:
            print(f"\nWARNING: Only {unique_desc}/{len(selected_indices)} unique descriptors!")
        
        # Save
        np.savez_compressed(
            output_path,
            xyz_world=selected_xyz,
            descriptors=selected_desc
        )
        
        print(f"\n✓ Saved pruned map to {output_path}")
        print(f"  Original:  {self.n_points} points")
        print(f"  Pruned:    {len(selected_indices)} points ({len(selected_indices)/self.n_points*100:.1f}%)")
        print(f"  Reduction: {(1 - len(selected_indices)/self.n_points)*100:.1f}%")


if __name__ == "__main__":
    import sys
    
    # Allow command line override for dataset
    dataset = sys.argv[1] if len(sys.argv) > 1 else "fr3"
    
    if dataset == "fr1":
        MAP_NPZ = 'resources/tum_fr1/map_databases/tumfr1_map_train.npz'
        COLMAP_DIR = 'resources/tum_fr1/project_files'
        OUTPUT_NPZ = 'resources/tum_fr1/map_databases/tumfr1_map_train_pruned50.npz'
        REDUCTION = 0.50
        B_COVER = 480  # 50% of original mean (963)
    else:  # fr3
        MAP_NPZ = 'resources/tum_fr3/map_databases/tumfr3_map_train.npz'
        COLMAP_DIR = 'resources/tum_fr3/project_files'
        OUTPUT_NPZ = 'resources/tum_fr3/map_databases/tumfr3_map_train_pruned40.npz'
        REDUCTION = 0.40
        B_COVER = 2000  # Conservative for FR3
    
    print("="*60)
    print(f"MAP PRUNING - {dataset.upper()}")
    print("="*60)
    
    # Load and initialize
    pruner = MapPruner(MAP_NPZ, COLMAP_DIR, b_cover=B_COVER)
    
    # Calculate target size
    target_k = int(pruner.n_points * (1 - REDUCTION))
    print(f"\nTarget: {target_k} points ({REDUCTION*100:.0f}% reduction)")
    
    # Sparse greedy selection
    selected = pruner.greedy_select_sparse(target_k)
    
    # Save
    pruner.save_pruned_map(selected, OUTPUT_NPZ)
    
    print("\n" + "="*60)
    print("DONE")
    print("="*60)
    print(f"\nNext steps:")
    print(f"1. Rebuild vocabulary tree with: aux_tasks/vocabTree.py")
    print(f"2. Test localization accuracy")
    print(f"3. Compare with original map")