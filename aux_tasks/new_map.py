from map_unc.map_build import MapBuilder
import glob
import os

# Get all COLMAP images (successfully reconstructed)
print("Loading COLMAP reconstruction...")
with open('resources/tum_fr2/project_files/images.txt', 'r') as f:
    colmap_images = []
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 10:
            colmap_images.append(parts[9])

colmap_set = set(colmap_images)
print(f"COLMAP successfully reconstructed {len(colmap_set)} images")

# Get filename order (chronological by timestamp)
print("\nGetting chronological order from filenames...")
all_frames = sorted(glob.glob('resources/tum_fr2/images/*.png'))
all_filenames = [os.path.basename(f) for f in all_frames]
print(f"Found {len(all_filenames)} total images in dataset")

print("\nSelecting first chronological images that were reconstructed...")
train_images = []
for fname in all_filenames:
    if fname in colmap_set:
        train_images.append(fname)
    if len(train_images) == 2200:
        break

train_images = set(train_images)

# Test images: remaining chronological images that were reconstructed
test_images = set()
for fname in all_filenames[len(train_images):]:
    if fname in colmap_set:
        test_images.add(fname)



# Build map using only train images
map_builder = MapBuilder()
map_3d_points, map_descriptors, map_track_lengths, map_ba_errors = map_builder.build_map_database(
    map_files='resources/tum_fr2/project_files',
    dataset_path='resources/tum_fr2/images',
    descriptors_path='resources/tum_fr2/project_files/descriptors',
    save_to='resources/tum_fr2/map_databases/tum_fr2_master.npz',
    train_images=train_images  # Only use train images
)