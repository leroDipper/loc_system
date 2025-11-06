import numpy as np
import cv2
from loc_modules import MapLoader
from loc_modules.feature_extractor import FeatureExtractor

# Load map
print("Loading map...")
data = np.load('resources/map_databases/colmap_map_train_set.npz')
map_descriptors = data['descriptors']

print(f"Map descriptors shape: {map_descriptors.shape}")
print(f"Map descriptors dtype: {map_descriptors.dtype}")
print(f"Map descriptors type: {type(map_descriptors)}")

# Extract features from a test image
extractor = FeatureExtractor()
dataset_root = 'colmap_database/large_map/large_set_test_640x480/'

keypoints, query_descriptors = extractor.resize_and_extract(dataset_root + 'frame_0143.jpg')

print(f"\nQuery descriptors shape: {query_descriptors.shape}")
print(f"Query descriptors dtype: {query_descriptors.dtype}")
print(f"Query descriptors type: {type(query_descriptors)}")

# Check if they match
print(f"\nDtypes match: {map_descriptors.dtype == query_descriptors.dtype}")
print(f"Columns match: {map_descriptors.shape[1] == query_descriptors.shape[1]}")
