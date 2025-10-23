from modules import MapBuilder

if __name__ == "__main__":
    # Build map database
    map_builder = MapBuilder()
    map_3d_points, map_descriptors = map_builder.build_map_database(
        map_files='/home/leroy-marewangepo/colmap_database/fig8/project_files',
        dataset_path='/home/leroy-marewangepo/colmap_database/fig8/train',
        descriptors_path='/home/leroy-marewangepo/colmap_database/fig8/descriptors_txt',
        save_to='/home/leroy-marewangepo/colmap_database/fig8/colmap_map.npz'
    )
    


