import numpy as np
from collections import defaultdict
import pandas as pd


class VocabTreeNode:
    def __init__(self, node_id, center, is_leaf, children):
        self.node_id = node_id
        self.center = center
        self.is_leaf = is_leaf
        self.children = children


def load_vocab_tree(filename):
    nodes = []

    with open(filename, "rb") as f:
        # ---- Header ----
        n_branches = np.fromfile(f, dtype=np.int32, count=1)[0]
        depth = np.fromfile(f, dtype=np.int32, count=1)[0]
        dim = np.fromfile(f, dtype=np.int32, count=1)[0]
        node_counter = np.fromfile(f, dtype=np.int32, count=1)[0]
        n_nodes = np.fromfile(f, dtype=np.int64, count=1)[0]

        print(f"Loading vocab tree:")
        print(f"  branches={n_branches}, depth={depth}, dim={dim}")
        print(f"  nodes={n_nodes}")

        # ---- Nodes ----
        for _ in range(n_nodes):
            center = np.fromfile(f, dtype=np.float32, count=dim)

            n_children = np.fromfile(f, dtype=np.int64, count=1)[0]
            children = np.fromfile(f, dtype=np.int32, count=n_children).tolist()

            is_leaf = bool(np.fromfile(f, dtype=np.uint8, count=1)[0])
            node_id = np.fromfile(f, dtype=np.int32, count=1)[0]

            nodes.append(
                VocabTreeNode(
                    node_id=node_id,
                    center=center,
                    is_leaf=is_leaf,
                    children=children
                )
            )

    return {
        "n_branches": n_branches,
        "depth": depth,
        "dim": dim,
        "nodes": nodes
    }


def query_vocab_tree(vocab_tree, descriptor):
    """
    Query the vocabulary tree with a single descriptor to find the leaf node ID.
    """
    current_node = vocab_tree['nodes'][0]  # Start at root

    while not current_node.is_leaf:
        # Find the child whose center is closest to the descriptor
        min_dist = float('inf')
        next_node = None

        for child_id in current_node.children:
            child_node = vocab_tree['nodes'][child_id]
            dist = np.linalg.norm(descriptor - child_node.center)

            if dist < min_dist:
                min_dist = dist
                next_node = child_node

        current_node = next_node

    return current_node.node_id




def compute_spatial_diversity(keypoints, k=5):
    """
    keypoints: (N, 2) array for ONE image
    returns: (N,) spatial diversity scores
    """
    N = keypoints.shape[0]
    diversity = np.zeros(N, dtype=np.float32)

    if N <= 1:
        return diversity

    # Pairwise distances
    dists = np.linalg.norm(
        keypoints[:, None, :] - keypoints[None, :, :],
        axis=2
    )

    for i in range(N):
        # Exclude self (distance 0)
        nearest = np.partition(dists[i], k+1)[1:k+1]
        diversity[i] = nearest.mean()

    return diversity


def normalize_per_image(values, image_ids):
    normed = np.zeros_like(values)

    for img_id in np.unique(image_ids):
        mask = image_ids == img_id
        v = values[mask]

        if len(v) > 0:
            normed[mask] = (v - v.min()) / (v.max() - v.min() + 1e-8)

    return normed

def save(npz_path, dataframe):
    """
    Save dataframe to npz file
    """
    print(f"\nSaving features to {npz_path}...")
    image_ids = dataframe['image_id'].to_numpy()
    image_names = dataframe['image_name'].to_numpy()
    keypoints = np.stack(dataframe['keypoint'].to_numpy())
    descriptors = np.stack(dataframe['descriptor'].to_numpy())
    feature_scores = dataframe['feature_scores'].to_numpy()
    
    np.savez_compressed(npz_path,
                        image_ids=image_ids,
                        image_names=image_names,
                        keypoints=keypoints,
                        descriptors=descriptors,
                        feature_scores=feature_scores)
    
    print("Save complete.")






if __name__ == "__main__":
    vocab_tree = load_vocab_tree('resources/tum_fr1/raw_feat/tum_fr1_features_tree.bin')
    print(f"Loaded vocabulary tree with {len(vocab_tree['nodes'])} nodes")

    # Example query
    data = np.load('resources/tum_fr1/raw_feat/tum_fr1_features_xfeat.npz',
                   allow_pickle=True)
    print(data.files)
    
    word_to_images = defaultdict(set)
    n_words = sum(1 for node in vocab_tree['nodes'] if node.is_leaf)
    assigned_words = []

    print("Querying vocabulary tree with descriptors...")
    for img_id, desc in zip(data['image_ids'], data['descriptors']):
        desc = desc.astype(np.float32)
        desc /= np.linalg.norm(desc) + 1e-8
        word_id = query_vocab_tree(vocab_tree, desc)
        assigned_words.append(word_id)
        word_to_images[word_id].add(img_id)

    word_usage = {wid: len(imgs) for wid, imgs in word_to_images.items()}

    print("Word usage stats:")
    print(f"  Used words: {len(word_usage)} / {n_words}")
    print(f"  Min images per word: {min(word_usage.values())}")
    print(f"  Max images per word: {max(word_usage.values())}")
    print(f"  Mean images per word: {np.mean(list(word_usage.values())):.2f}")


    IDF = {}
    N_total_images = len(set(data['image_ids']))


    leaf_ids = [node.node_id for node in vocab_tree['nodes'] if node.is_leaf]
    print("Computing IDF values...")
    for word_id in leaf_ids:
        n_images_with_word = len(word_to_images[word_id])
        IDF[word_id] = np.log(N_total_images / (n_images_with_word + 1e-8))

    idf_values = np.array(list(IDF.values()))

    print("IDF stats:")
    print(f"  min IDF: {idf_values.min():.4f}")
    print(f"  max IDF: {idf_values.max():.4f}")
    print(f"  mean IDF: {idf_values.mean():.4f}")

    idf_min = idf_values.min()
    idf_max = idf_values.max()

    IDF_norm = {
        w: (v - idf_min) / (idf_max - idf_min + 1e-8)
        for w, v in IDF.items()
    }

    detector_norm = normalize_per_image(
    data['detector_scores'],
    data['image_ids']
    )

    image_ids = data['image_ids']
    keypoints = data['keypoints']
    image_names = data['image_names']
    descriptors = data['descriptors']

    spatial_diversity = np.zeros(len(keypoints), dtype=np.float32)

    for img_id in np.unique(image_ids):
        mask = image_ids == img_id
        kp = keypoints[mask][:, :2]  # (x, y)

        div = compute_spatial_diversity(kp, k=5)
        spatial_diversity[mask] = div

    spatial_div_norm = normalize_per_image(spatial_diversity, image_ids)

    alpha = 0.0  # IDF
    beta  = 1.0   # Detector strength
    gamma = 0.0   # Spatial diversity

    combined_scores = np.zeros(len(descriptors), dtype=np.float32)

    print("Assigning combined scores...")
    for i, desc in enumerate(descriptors):
        desc = desc.astype(np.float32)
        desc /= np.linalg.norm(desc) + 1e-8

        # word_id = query_vocab_tree(vocab_tree, desc)
        word_id = assigned_words[i]
        idf_score = IDF_norm.get(word_id, 0.0)
        det_score = detector_norm[i]
        div_score = spatial_div_norm[i]

        combined_scores[i] = (
            alpha * idf_score +
            beta  * det_score +
            gamma * div_score
        )




    print("Combined score stats:")
    print(f"  min: {combined_scores.min():.4f}")
    print(f"  max: {combined_scores.max():.4f}")
    print(f"  mean: {combined_scores.mean():.4f}")

    # # Check per-image top features
    # for img_id in np.unique(image_ids)[:10]:
    #     mask = image_ids == img_id
    #     scores = combined_scores[mask]
    #     print(f"Image {img_id}: top score = {scores.max():.4f}, mean = {scores.mean():.4f}")

    print("Pruning features to top 600 per image...")
    top_feat = 600
    pruned_data = []

    for img_id in np.unique(image_ids):
        mask = image_ids == img_id

        scores = combined_scores[mask]
        img_indices = np.where(mask)[0]

        # sort DESCENDING (best first)
        order = np.argsort(scores)[::-1][:top_feat]
        selected = img_indices[order]

        for i in selected:
            pruned_data.append({
                "image_id": image_ids[i],
                "image_name": image_names[i],
                "keypoint": keypoints[i],
                "descriptor": descriptors[i],
                "feature_scores": combined_scores[i]
            })


    df = pd.DataFrame(pruned_data)
    #df = df.sort_values(by=["image_name", "feature_scores"], ascending=[True, False])

    unique_names = df["image_name"].unique()

    print(unique_names[:10])
    print(unique_names[-10:])

    print(f"Pruned dataset has {len(df)} features.")
    print(f"Sample rows:")
    print(df.head())

    # Save pruned features
    save('resources/tum_fr1/map_databases/tum_fr1_features_pruned.npz', df)


    

        
        



        









