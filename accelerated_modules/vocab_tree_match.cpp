#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <pybind11/stl.h>
#include <fstream>  // for std::ifstream
#include <iostream> // for std::cout

namespace py = pybind11;

class VocabTreeMatcher {
    private:

        struct TreeNode {
        std::vector<float> center;
        std::vector<int> children;
        bool is_leaf;
        int node_id;
        };

        float compute_distance(const std::vector<float>& a, const std::vector<float>& b) {
            float dist = 0.0f;
            for (size_t i = 0; i < a.size(); i++) {
                float diff = a[i] - b[i];
                dist += diff * diff;
            }
            return std::sqrt(dist);
        }
        std::vector<std::vector<float>> map_descriptors;
        std::vector<std::vector<int>> word_groups;
        int n_branches;
        int depth;
        int dim;
        int node_counter;
        std::vector<TreeNode> nodes;
        int top_k;

        std::vector<std::vector<float>> get_vocabulary() {
            std::vector<std::vector<float>> vocabulary;
            for (const auto& node : nodes) {
                if (node.is_leaf) {
                vocabulary.push_back(node.center);
                }
            }
            return vocabulary;
        }

        void load(const std::string& filename) {
            std::ifstream file(filename, std::ios::binary);
            if (!file) {
                throw std::runtime_error("Cannot open file for reading: " + filename);
            }

            // read metadata
            file.read(reinterpret_cast<char*>(&n_branches), sizeof(n_branches));
            file.read(reinterpret_cast<char*>(&depth), sizeof(depth));
            file.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            file.read(reinterpret_cast<char*>(&node_counter), sizeof(node_counter));

            // read number of nodes
            size_t num_nodes = nodes.size();
            file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));

            // Clear existing nodes
            nodes.clear();
            nodes.reserve(num_nodes);

            for (size_t i = 0; i < num_nodes; i++) {
                TreeNode node;
                // Read center
                node.center.resize(dim);
                file.read(reinterpret_cast<char*>(node.center.data()), 
                        dim * sizeof(float));

                // Read children
                size_t num_children;
                file.read(reinterpret_cast<char*>(&num_children), sizeof(num_children));
                node.children.resize(num_children);
                file.read(reinterpret_cast<char*>(node.children.data()), 
                        num_children * sizeof(int));
                
                // Read is_leaf and node_id
                file.read(reinterpret_cast<char*>(&node.is_leaf), sizeof(node.is_leaf));
                file.read(reinterpret_cast<char*>(&node.node_id), sizeof(node.node_id));
                
                nodes.push_back(node);
            }

            file.close();
            std::cout << "Vocabulary tree loaded from " << filename << std::endl;
            std::cout << "Tree has " << nodes.size() << " nodes." << std::endl;

            auto vocab = get_vocabulary();
            std::cout << "Vocabulary size: " << vocab.size() << std::endl;

    }

    int query_tree(const std::vector<float>& descriptor) {
        int current_node_idx = 0; // Start at root
        for (int d = 0; d < depth; d++) { 
            const TreeNode& current_node = nodes[current_node_idx];
            if (current_node.is_leaf) {
                break; // reached leaf
            }

            int best_child = -1;
            float best_dist = std::numeric_limits<float>::max();
            for (int child_id : current_node.children) {
                const TreeNode& child_node = nodes[child_id];
                float dist = 0.0f;
                for (int i = 0; i < dim; i++) {
                    float diff = descriptor[i] - child_node.center[i];
                    dist = dist + diff*diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_child = child_id;
                }
            }
            if (best_child == -1) {
                break; // no children found
            }
            current_node_idx = best_child;
        }
        return nodes[current_node_idx].node_id;
    }

    void build_inverted_index() {
        word_groups.resize(node_counter);
        for (size_t i = 0; i < map_descriptors.size(); i++) {
            const auto& descriptor = map_descriptors[i];
            int word_id = query_tree(descriptor);
            word_groups[word_id].push_back(i);
        }
    }

    std::vector<int> find_k_nearest_leaves(const std::vector<float>& descriptor, int k) {
    std::vector<std::pair<float, int>> leaf_distances;

    for (const auto& node : nodes) {
        if (node.is_leaf) {
            float dist = compute_distance(descriptor, node.center);
            leaf_distances.emplace_back(dist, node.node_id);
        }

    }

    std::partial_sort(leaf_distances.begin(), 
                      leaf_distances.begin() + std::min(k, (int)leaf_distances.size()), 
                      leaf_distances.end());

    std::vector<int> nearest_leaves;
    for (int i = 0; i < std::min(k, (int)leaf_distances.size()); i++) {
        nearest_leaves.push_back(leaf_distances[i].second);
    }

    return nearest_leaves;
    }


    public:
        VocabTreeMatcher(const std::string& vocab_filename,
                 py::array_t<float> map_desc_np) {
    
        load(vocab_filename);
        
        // Copy map descriptors FIRST
        auto map_buf = map_desc_np.request();
        int n_map = map_buf.shape[0];
        // Note: dim is already set by load(), so don't override it
        
        map_descriptors.resize(n_map);
        float* map_ptr = (float*)map_buf.ptr;
        for (int i = 0; i < n_map; i++) {
            map_descriptors[i].resize(dim);
            for (int j = 0; j < dim; j++) {
                map_descriptors[i][j] = map_ptr[i * dim + j];
            }
        }
        
        // THEN build inverted index (now map_descriptors is populated)
        build_inverted_index();
    }

    py::tuple match(py::array_t<float> query_desc_np, float ratio_threshold = 0.80, int k_nearest_words = 1) {

        auto query_buf = query_desc_np.request();
        int n_query = query_buf.shape[0];
        float* query_ptr = (float*)query_buf.ptr;
        
        std::vector<int> match_query_idx;
        std::vector<int> match_map_idx;
        std::vector<float> match_distances;

        for (int query_idx = 0; query_idx < n_query; query_idx++) {
            std::vector<float> query_descriptor(dim);
            for (int j = 0; j < dim; j++) {
                query_descriptor[j] = query_ptr[query_idx * dim + j];
            }



            std::vector<int> candidate_indices;

            if (k_nearest_words == 1) {
                int leaf_node_id = query_tree(query_descriptor);
                candidate_indices = word_groups[leaf_node_id];
            }
            else {
                auto nearest_leaves = find_k_nearest_leaves(query_descriptor, k_nearest_words);
                std::unordered_set<int> unique_candidates;

                for (int leaf_id : nearest_leaves) {
                    for (int idx : word_groups[leaf_id]) {
                        unique_candidates.insert(idx);
                    }
                }

                candidate_indices.assign(unique_candidates.begin(), unique_candidates.end());

            }


            // Check if we have candidates
            if (candidate_indices.size() < 2) continue;  // Need at least 2 for ratio test


            // Find best and second-best among candidates
            float best_dist = std::numeric_limits<float>::max();
            float second_best_dist = std::numeric_limits<float>::max();
            int best_idx = -1;
            
            for (int map_idx : candidate_indices) {
                float dist = compute_distance(query_descriptor, map_descriptors[map_idx]);
                
                if (dist < best_dist) {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx = map_idx;
                } else if (dist < second_best_dist) {
                    second_best_dist = dist;
                }
            }

            // Lowe's ratio test
            if (best_idx >= 0 && best_dist < ratio_threshold * second_best_dist) {
                match_query_idx.push_back(query_idx);
                match_map_idx.push_back(best_idx);
                match_distances.push_back(best_dist);
            }

        }

        // Convert results to numpy arrays
        py::array_t<int> query_idx_out(match_query_idx.size());
        py::array_t<int> map_idx_out(match_map_idx.size());
        py::array_t<float> distances_out(match_distances.size());

        auto q_ptr = query_idx_out.mutable_unchecked<1>();
        auto m_ptr = map_idx_out.mutable_unchecked<1>();
        auto d_ptr = distances_out.mutable_unchecked<1>();

        for (size_t i = 0; i < match_query_idx.size(); i++) {
            q_ptr(i) = match_query_idx[i];
            m_ptr(i) = match_map_idx[i];
            d_ptr(i) = match_distances[i];
        }

        // TODO: For each query descriptor:
        //   1. Convert to std::vector<float>
        //   2. Find its leaf node using query_tree()
        //   3. Get candidates from word_groups[leaf_node_id]
        //   4. Find best and second-best matches among candidates
        //   5. Apply Lowe's ratio test
        //   6. Store matches
        
        // TODO: Convert results to numpy arrays and return
        
        return py::make_tuple(query_idx_out, map_idx_out, distances_out);
}

};

PYBIND11_MODULE(vocab_tree_match, m) {
    py::class_<VocabTreeMatcher>(m, "VocabTreeMatcher")
        .def(py::init<const std::string&, py::array_t<float>>())
        .def("match", &VocabTreeMatcher::match,
             py::arg("query_desc_np"), py::arg("ratio_threshold") = 0.80,
             py::arg("k_nearest_words") = 1);
}








