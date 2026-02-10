from map_unc.map_build import MapBuilder

# Get COLMAP images in order
with open('resources/mh_01/proj_files/images.txt', 'r') as f:
    colmap_images = []
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 10:
            colmap_images.append(parts[9])

# Split into train/test
train_images = set(colmap_images[:1100]) 
test_images = set(colmap_images[1100:])   

print(f"Train: {len(train_images)} images")
print(f"Test: {len(test_images)} images")

# Build map using only train images
map_builder = MapBuilder()
map_3d_points, map_descriptors, map_track_lengths, map_ba_errors = map_builder.build_map_database(
    map_files='resources/mh_01/proj_files',
    dataset_path='resources/mh_01/images',
    descriptors_path='resources/mh_01/proj_files/descriptors',
    save_to='resources/mh_01/map_databases/mh_01_master.npz',
    train_images=train_images  # Only use train images
)