import numpy as np

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


if __name__ == "__main__":
    vocab_tree = load_vocab_tree('resources/tum_fr3/vocabularies/vocab_tree_40_pruned.bin')
    print(f"Loaded vocabulary tree with {len(vocab_tree['nodes'])} nodes")

    # Example query
    example_descriptor = np.random.rand(vocab_tree['dim']).astype(np.float32)
    leaf_node_id = query_vocab_tree(vocab_tree, example_descriptor)
    print(f"Example descriptor assigned to leaf node ID: {leaf_node_id}")


