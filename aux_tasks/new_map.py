from loc_modules.map_builder import MapBuilder

# Get COLMAP images in order
with open('resources/tum_fr1/project_files/images.txt', 'r') as f:
    colmap_images = []
    for line in f:
        if line.startswith('#') or not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) == 10:
            colmap_images.append(parts[9])

# Split into train/test
train_images = set(colmap_images[:700]) 
test_images = set(colmap_images[700:])   

print(f"Train: {len(train_images)} images")
print(f"Test: {len(test_images)} images")

# Build map using only train images
map_builder = MapBuilder()
map_3d_points, map_descriptors = map_builder.build_map_database(
    map_files='resources/tum_fr1/project_files',
    dataset_path='resources/tum_fr1/images',
    descriptors_path='resources/tum_fr1/project_files/descriptors',
    save_to='resources/tum_fr1/map_databases/tum_fr1_train.npz',
    train_images=train_images  # Only use train images
)