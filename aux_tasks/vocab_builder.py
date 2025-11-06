from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":

    data = np.load('resources/map_databases/colmap_map_train_set.npz')
    map_3d_points = data['xyz_world']
    map_descriptors = data['descriptors']
    print(f"Loaded map: {len(map_3d_points)} points")

    k = 1000
    iters = 120
    
    print(f"Running k-means clustering with k={k}")
    
    # sklearn's KMeans with verbose output
    kmeans_model = KMeans(n_clusters=k, max_iter=iters, verbose=1, n_init=1)
    kmeans_model.fit(map_descriptors)
    
    vocabulary = kmeans_model.cluster_centers_
    variance = kmeans_model.inertia_
    
    np.save('resources/vocabularies/vocabulary_circ_f8.npy', vocabulary)
    print(f"Vocabulary saved. Final inertia: {variance:.4f}")