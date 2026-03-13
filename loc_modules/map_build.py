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
                    error = float(parts[7])  # BA reprojection error

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
                        'track': track,
                        'track_length': len(track),
                        'ba_error': error
                    })

        print(f"Loaded {len(points)} 3D points")
        return points

    def load_image_ids_and_descriptors(self, dataset_path, descriptors_path, train_images=None):
        """
        Load descriptors for images.

        Args:
            dataset_path: Path to images directory
            descriptors_path: Path to descriptors directory
            train_images: Optional set of image names to filter for (train set only)
        """
        dataset_path = Path(dataset_path)
        descriptors_path = Path(descriptors_path)

        data = []
        image_files = list(dataset_path.glob("*.jpg")) + list(dataset_path.glob("*.png"))
        image_files = sorted(image_files)

        if train_images is not None:
            image_files = [f for f in image_files if f.name in train_images]
            print(f"Using {len(image_files)} train images for map building")

        for img_path in image_files:
            desc_path = descriptors_path / f"{img_path.name}_desc.txt"
            if not desc_path.exists():
                print(f"Warning: Missing descriptor for {img_path.name}")
                continue

            descriptors = []
            with open(desc_path, 'r') as f:
                for line in f:
                    if line.strip():
                        descriptor_array = np.array([int(x) for x in line.split()])
                        descriptors.append(descriptor_array)

            data.append({
                'image': img_path.name,
                'descriptors': np.array(descriptors)
            })

        df = pd.DataFrame(data)
        print(f"Loaded {len(data)} images with descriptors")
        return df

    def build_map_database(self, map_files, dataset_path, descriptors_path, save_to=None, train_images=None):
        """
        Build map database from COLMAP outputs.

        Args:
            train_images: Optional set of image names to use for building map (excludes test set)
        """
        points = self.load_map_data(map_files)
        df = self.load_image_ids_and_descriptors(dataset_path, descriptors_path, train_images)
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

        map_3d_points = []
        map_descriptors = []
        map_track_lengths = []
        map_ba_errors = []
        map_image_ids = []       # sequential index per map point
        image_name_to_idx = {}   # image_name -> sequential index
        image_names_ordered = [] # ordered list of unique image names

        skipped_test_images = 0

        for point in points:
            if not point['track']:
                continue

            first_img_id, first_kp_idx = point['track'][0]
            img_name = images_data.get(first_img_id)

            if not img_name:
                continue

            if train_images is not None and img_name not in train_images:
                skipped_test_images += 1
                continue

            img_row = df[df['image'] == img_name]
            if img_row.empty:
                continue

            descriptors = img_row.iloc[0]['descriptors']
            if first_kp_idx < len(descriptors):
                # Assign sequential image index
                if img_name not in image_name_to_idx:
                    image_name_to_idx[img_name] = len(image_names_ordered)
                    image_names_ordered.append(img_name)

                map_3d_points.append(point['xyz'])
                map_descriptors.append(descriptors[first_kp_idx])
                map_track_lengths.append(point['track_length'])
                map_ba_errors.append(point['ba_error'])
                map_image_ids.append(image_name_to_idx[img_name])

        map_3d_points = np.array(map_3d_points, dtype=np.float32)
        map_descriptors = np.array(map_descriptors, dtype=np.float32)
        map_track_lengths = np.array(map_track_lengths, dtype=np.int32)
        map_ba_errors = np.array(map_ba_errors, dtype=np.float32)
        map_image_ids = np.array(map_image_ids, dtype=np.int32)

        print(f"\nBuilt map with {len(map_3d_points)} points")
        if train_images:
            print(f"Skipped {skipped_test_images} points from test images")
        print(f"3D points shape: {map_3d_points.shape}")
        print(f"Descriptors shape: {map_descriptors.shape}")
        print(f"Unique training images: {len(image_names_ordered)}")

        print(f"\nMap quality statistics:")
        print(f"  Track length - mean: {np.mean(map_track_lengths):.1f}, "
              f"median: {np.median(map_track_lengths):.0f}, "
              f"min: {np.min(map_track_lengths)}, "
              f"max: {np.max(map_track_lengths)}")
        print(f"  BA error - mean: {np.mean(map_ba_errors):.4f}, "
              f"median: {np.median(map_ba_errors):.4f}, "
              f"min: {np.min(map_ba_errors):.4f}, "
              f"max: {np.max(map_ba_errors):.4f}")
        high_quality_count = np.sum(map_track_lengths >= 5)
        print(f"  High quality points (track >= 5): {high_quality_count} "
              f"({high_quality_count/len(map_track_lengths)*100:.1f}%)")

        if save_to:
            self.save_map(save_to, map_3d_points, map_descriptors,
                         map_track_lengths, map_ba_errors,
                         map_image_ids, image_names_ordered)

        return map_3d_points, map_descriptors, map_track_lengths, map_ba_errors

    def save_map(self, npz_path, map_3d_points, map_descriptors,
                 map_track_lengths, map_ba_errors,
                 map_image_ids, image_names_ordered):
        """Save map with quality metrics and image provenance."""
        np.savez_compressed(
            npz_path,
            xyz_world=map_3d_points,
            descriptors=map_descriptors,
            track_lengths=map_track_lengths,
            ba_errors=map_ba_errors,
            image_ids=map_image_ids,
            image_names=np.array(image_names_ordered)
        )
        print(f"Saved map with quality metrics to {npz_path}")