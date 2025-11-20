from loc_modules import MapBuilder

if __name__ == "__main__":
    # Configuration
    N_TRAIN_IMAGES =  500 # Use first 2000 images for map building
    
    # Build map database
    map_builder = MapBuilder()
    map_3d_points, map_descriptors = map_builder.build_map_database(
        map_files='sift_stuff/tum_fr1/project_files',
        dataset_path='resources/tum_fr1/images',
        descriptors_path='sift_stuff/tum_fr1/project_files/descriptors',
        save_to='sift_stuff/tum_fr1/map_databases/tumfr1_map_train.npz',
        max_images=N_TRAIN_IMAGES  # Add this parameter
    )

