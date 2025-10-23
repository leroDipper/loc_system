from modules import MapBuilder

if __name__ == "__main__":
    # Build map database
    map_builder = MapBuilder()
    map_3d_points, map_descriptors = map_builder.build_map_database(
        map_files='/colmap_database/large_map/project_files',
        dataset_path='colmap_database/large_map/large_set_train',
        descriptors_path='colmap_database/large_map/descriptors_txt',
        save_to='colmap_database/large_map/colmap_map.npz'
    )
    


