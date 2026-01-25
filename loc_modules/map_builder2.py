from pathlib import Path
import numpy as np
import pandas as pd

class MapBuilder:
    """Builds a 3D map and descriptor database from COLMAP outputs."""

    def load_map_data(self, map_files):
        map_path = Path(map_files)
        points_3d = map_path / "points3D.txt"

        if not points_3d.exists():
            raise FileNotFoundError(f"{points_3d} not found")

        points = []
        with open(points_3d, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) >= 8:
                    point_id = int(parts[0])
                    x, y, z = map(float, parts[1:4])
                    error = float(parts[7])

                    # Parse (IMAGE_ID, POINT2D_IDX) pairs
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

        print(f"Loaded {len(points)} 3D points")
        return points

    def load_image_ids_and_descriptors(self, dataset_path, float_descriptors_path, max_images=None):
        """
        Load float32 L2-normalized descriptors from NPZ file.
        
        Args:
            dataset_path: Path to image directory (for getting image list)
            float_descriptors_path: Path to NPZ file with float descriptors
            max_images: Optional limit on number of images to load
        """
        dataset_path = Path(dataset_path)
        
        # Load float descriptors from NPZ
        float_data = np.load(float_descriptors_path)
        
        data = []
        image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        image_files = sorted(image_files)
        
        # Limit to first N images if specified
        if max_images is not None:
            image_files = image_files[:max_images]
            print(f"Using first {max_images} images for map building")
        
        for img_path in image_files:
            img_name = img_path.name
            
            # Check if descriptors exist in NPZ
            if img_name not in float_data:
                print(f"Warning: Missing float descriptors for {img_name}")
                continue
            
            descriptors = float_data[img_name]
            
            data.append({
                'image': img_name,
                'descriptors': descriptors
            })
        
        df = pd.DataFrame(data)
        print(f"Loaded {len(data)} images with float32 descriptors")
        return df

    def build_map_database(self, map_files, dataset_path, float_descriptors_path, save_to=None, max_images=None):
        points = self.load_map_data(map_files)
        df = self.load_image_ids_and_descriptors(dataset_path, float_descriptors_path, max_images)
        map_path = Path(map_files)

        images_data = {}
        with open(map_path / "images.txt", 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue

                parts = line.strip().split()
                if len(parts) == 10:
                    img_id = int(parts[0])
                    img_name = parts[9]
                    images_data[img_id] = img_name

        print(f"Loaded {len(images_data)} image mappings")

        map_3d_points, map_descriptors = [], []
        for point in points:
            if not point['track']:
                continue

            first_img_id, first_kp_idx = point['track'][0]
            img_name = images_data.get(first_img_id)

            if not img_name:
                continue

            img_row = df[df['image'] == img_name]
            if img_row.empty:
                continue

            descriptors = img_row.iloc[0]['descriptors']
            if first_kp_idx < len(descriptors):
                map_3d_points.append(point['xyz'])
                map_descriptors.append(descriptors[first_kp_idx])

        map_3d_points = np.array(map_3d_points, dtype=np.float32)
        map_descriptors = np.array(map_descriptors, dtype=np.float32)

        print(f"\nBuilt map with {len(map_3d_points)} points")
        print(f"3D points shape: {map_3d_points.shape}")
        print(f"Descriptors shape: {map_descriptors.shape}")

        if save_to:
            self.save_map(save_to, map_3d_points, map_descriptors)

        return map_3d_points, map_descriptors

    def save_map(self, npz_path, map_3d_points, map_descriptors):
        np.savez_compressed(npz_path, xyz_world=map_3d_points, descriptors=map_descriptors)
        print(f"Saved map to {npz_path}")