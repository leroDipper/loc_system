import numpy as np
from sklearn.cluster import KMeans



class VocabTreeNode:
    """Node in the vocabulary tree."""
    def __init__(self, node_id, center, is_leaf=False):
        self.node_id = node_id
        self.center = center
        self.children = []
        self.is_leaf = is_leaf
    
    def add_child(self, child_id):
        self.children.append(child_id)


class VocabTreeBuilder:
    """Build a hierarchical vocabulary tree using recursive k-means."""
    
    def __init__(self, n_branches, depth, descriptor_dim):
        self.n_branches = n_branches
        self.depth = depth
        self.dim = descriptor_dim
        self.nodes = []
        self.node_counter = 0
    
    def _create_node(self, center, is_leaf=False):
        node = VocabTreeNode(self.node_counter, center, is_leaf)
        self.nodes.append(node)
        self.node_counter += 1
        return node.node_id
    
    def _build_recursive(self, descriptors, current_depth):
        n_descriptors = len(descriptors)
        center = np.mean(descriptors, axis=0)
        is_leaf = (current_depth >= self.depth) or (n_descriptors < self.n_branches)
        
        node_id = self._create_node(center, is_leaf)
        
        if is_leaf:
            return node_id
        
        kmeans = KMeans(n_clusters=self.n_branches, max_iter=20, n_init=1, verbose=0)
        labels = kmeans.fit_predict(descriptors)
        
        for cluster_id in range(self.n_branches):
            cluster_mask = labels == cluster_id
            cluster_descriptors = descriptors[cluster_mask]
            
            if len(cluster_descriptors) > 0:
                child_id = self._build_recursive(cluster_descriptors, current_depth + 1)
                self.nodes[node_id].add_child(child_id)
        
        return node_id
    
    def build(self, descriptors):
        self._build_recursive(descriptors, current_depth=0)
        n_leaves = sum(1 for node in self.nodes if node.is_leaf)
        print(f"Built tree: {len(self.nodes)} nodes, {n_leaves} leaves")
    
    def get_vocabulary(self):
        return np.array([node.center for node in self.nodes if node.is_leaf])
    
    def save(self, filename):
        """Save tree in binary format that C++ VocabTree can load."""
        with open(filename, 'wb') as f:
            # Write metadata
            np.array([self.n_branches], dtype=np.int32).tofile(f)
            np.array([self.depth], dtype=np.int32).tofile(f)
            np.array([self.dim], dtype=np.int32).tofile(f)
            np.array([self.node_counter], dtype=np.int32).tofile(f)
            
            # Write number of nodes
            np.array([len(self.nodes)], dtype=np.int64).tofile(f)
            
            # Write each node
            for node in self.nodes:
                # Write center
                np.array(node.center, dtype=np.float32).tofile(f)
                
                # Write children
                np.array([len(node.children)], dtype=np.int64).tofile(f)
                np.array(node.children, dtype=np.int32).tofile(f)
                
                # Write is_leaf and node_id
                np.array([node.is_leaf], dtype=np.uint8).tofile(f)
                np.array([node.node_id], dtype=np.int32).tofile(f)


       


if __name__ == "__main__":
    data = np.load('resources/mh_01/map_databases/mh_01_train.npz')

    map_descriptors = data['descriptors'].astype(np.float32)
    
    n_branches = 10
    depth = 4
    descriptor_dim = map_descriptors.shape[1]
    
    builder = VocabTreeBuilder(n_branches, depth, descriptor_dim)
    builder.build(map_descriptors)
    builder.save('resources/mh_01/vocabularies/vocab_tree.bin')
    
    vocab = builder.get_vocabulary()
    print(f"Vocabulary: {vocab.shape}")