from loc_modules import MapBuilder

if __name__ == "__main__":
    # Build map database
    map_builder = MapBuilder()
    map_3d_points, map_descriptors = map_builder.build_map_database(
        map_files='colmap_database/large_map_xfeat/project_files',
        dataset_path='colmap_database/large_map/large_set_train_640x480',
        descriptors_path='colmap_database/large_map_xfeat/descriptors_xfeat_640x480',
        save_to='resources/map_databases/colmap_map_train_set.npz'
    )
    


