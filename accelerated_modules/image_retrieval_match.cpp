#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>
#include <fstream>
#include <iostream>
#include <queue>

namespace py = pybind11;

class ImageRetrievalMatcher {
private:
    struct TreeNode {
        std::vector<float> centre;
        std::vector<int> children;
        bool is_leaf;
        int node_id;
    };

    int n_branches, depth, dim, node_counter;
    std::vector<TreeNode> nodes;

    // word_id -> list of map point indices
    std::vector<std::vector<int>> word_to_points;
    // word_id -> list of image indices (may contain duplicates — that's fine, counts as votes)
    std::vector<std::vector<int>> word_to_images;

    std::vector<std::vector<float>> map_descriptors;
    std::vector<int> map_image_ids; // image index per map point
    int n_images;

    float compute_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float dist = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }

    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) throw std::runtime_error("Cannot open vocabulary: " + filename);

        file.read(reinterpret_cast<char*>(&n_branches),   sizeof(n_branches));
        file.read(reinterpret_cast<char*>(&depth),        sizeof(depth));
        file.read(reinterpret_cast<char*>(&dim),          sizeof(dim));
        file.read(reinterpret_cast<char*>(&node_counter), sizeof(node_counter));

        size_t num_nodes = 0;
        file.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
        nodes.clear();
        nodes.reserve(num_nodes);

        for (size_t i = 0; i < num_nodes; i++) {
            TreeNode node;
            node.centre.resize(dim);
            file.read(reinterpret_cast<char*>(node.centre.data()), dim * sizeof(float));

            size_t num_children;
            file.read(reinterpret_cast<char*>(&num_children), sizeof(num_children));
            node.children.resize(num_children);
            file.read(reinterpret_cast<char*>(node.children.data()), num_children * sizeof(int));

            file.read(reinterpret_cast<char*>(&node.is_leaf),  sizeof(node.is_leaf));
            file.read(reinterpret_cast<char*>(&node.node_id),  sizeof(node.node_id));
            nodes.push_back(node);
        }

        std::cout << "Vocabulary loaded: " << nodes.size() << " nodes" << std::endl;
    }

    int query_tree(const std::vector<float>& descriptor) {
        int current = 0;
        for (int d = 0; d < depth; d++) {
            const TreeNode& node = nodes[current];
            if (node.is_leaf) break;

            int best_child = -1;
            float best_dist = std::numeric_limits<float>::max();
            for (int child_id : node.children) {
                float dist = 0.0f;
                for (int i = 0; i < dim; i++) {
                    float diff = descriptor[i] - nodes[child_id].centre[i];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best_child = child_id; }
            }
            if (best_child == -1) break;
            current = best_child;
        }
        return nodes[current].node_id;
    }

    void build_indices() {
        word_to_points.resize(node_counter);
        word_to_images.resize(node_counter);

        for (int i = 0; i < (int)map_descriptors.size(); i++) {
            int word_id = query_tree(map_descriptors[i]);
            word_to_points[word_id].push_back(i);
            word_to_images[word_id].push_back(map_image_ids[i]);
        }

        std::cout << "Inverted index built over " << map_descriptors.size()
                  << " points, " << n_images << " images" << std::endl;
    }

public:
    ImageRetrievalMatcher(const std::string& vocab_filename,
                          py::array_t<float> map_desc_np,
                          py::array_t<int>   image_ids_np,
                          int n_images_total) : n_images(n_images_total) {
        load(vocab_filename);

        auto map_buf = map_desc_np.request();
        auto ids_buf = image_ids_np.request();
        int n_map = map_buf.shape[0];

        float* map_ptr = (float*)map_buf.ptr;
        int*   ids_ptr = (int*)ids_buf.ptr;

        map_descriptors.resize(n_map);
        map_image_ids.resize(n_map);
        for (int i = 0; i < n_map; i++) {
            map_descriptors[i].resize(dim);
            for (int j = 0; j < dim; j++)
                map_descriptors[i][j] = map_ptr[i * dim + j];
            map_image_ids[i] = ids_ptr[i];
        }

        build_indices();
    }

    // Returns (query_idx, map_idx, distances) — same format as vocab_tree_match.
    py::tuple match(py::array_t<float> query_desc_np,
                    float ratio_threshold = 0.80f,
                    int top_k_images = 10) {

        auto query_buf = query_desc_np.request();
        int n_query = query_buf.shape[0];
        float* query_ptr = (float*)query_buf.ptr;

        // ---- Step 1: vote for images ----
        std::vector<int> image_votes(n_images, 0);

        for (int qi = 0; qi < n_query; qi++) {
            std::vector<float> qdesc(dim);
            for (int j = 0; j < dim; j++)
                qdesc[j] = query_ptr[qi * dim + j];

            int word_id = query_tree(qdesc);
            // Each image that has a point in this leaf gets one vote
            for (int img_idx : word_to_images[word_id])
                image_votes[img_idx]++;
        }

        // ---- Step 2: pick top-K images ----
        // Partial sort — only need top_k_images
        std::vector<int> image_order(n_images);
        for (int i = 0; i < n_images; i++) image_order[i] = i;
        int k = std::min(top_k_images, n_images);
        std::partial_sort(image_order.begin(), image_order.begin() + k, image_order.end(),
                          [&](int a, int b){ return image_votes[a] > image_votes[b]; });

        std::unordered_set<int> top_image_set(image_order.begin(), image_order.begin() + k);

        // ---- Step 3: collect candidate map points from top-K images ----
        std::vector<int> candidates;
        candidates.reserve(n_query * 50); // rough upper bound
        for (int i = 0; i < (int)map_descriptors.size(); i++) {
            if (top_image_set.count(map_image_ids[i]))
                candidates.push_back(i);
        }

        // ---- Step 4: brute-force match each query against candidates ----
        std::vector<int>   match_query_idx;
        std::vector<int>   match_map_idx;
        std::vector<float> match_distances;

        for (int qi = 0; qi < n_query; qi++) {
            std::vector<float> qdesc(dim);
            for (int j = 0; j < dim; j++)
                qdesc[j] = query_ptr[qi * dim + j];

            if (candidates.size() < 2) continue;

            float best_dist        = std::numeric_limits<float>::max();
            float second_best_dist = std::numeric_limits<float>::max();
            int   best_idx         = -1;

            for (int map_idx : candidates) {
                float dist = compute_distance(qdesc, map_descriptors[map_idx]);
                if (dist < best_dist) {
                    second_best_dist = best_dist;
                    best_dist = dist;
                    best_idx  = map_idx;
                } else if (dist < second_best_dist) {
                    second_best_dist = dist;
                }
            }

            if (best_idx >= 0 && best_dist < ratio_threshold * second_best_dist) {
                match_query_idx.push_back(qi);
                match_map_idx.push_back(best_idx);
                match_distances.push_back(best_dist);
            }
        }

        // ---- Return ----
        py::array_t<int>   query_idx_out(match_query_idx.size());
        py::array_t<int>   map_idx_out(match_map_idx.size());
        py::array_t<float> distances_out(match_distances.size());

        auto q_ptr = query_idx_out.mutable_unchecked<1>();
        auto m_ptr = map_idx_out.mutable_unchecked<1>();
        auto d_ptr = distances_out.mutable_unchecked<1>();

        for (size_t i = 0; i < match_query_idx.size(); i++) {
            q_ptr(i) = match_query_idx[i];
            m_ptr(i) = match_map_idx[i];
            d_ptr(i) = match_distances[i];
        }

        return py::make_tuple(query_idx_out, map_idx_out, distances_out);
    }
};

PYBIND11_MODULE(image_retrieval_match, m) {
    py::class_<ImageRetrievalMatcher>(m, "ImageRetrievalMatcher")
        .def(py::init<const std::string&, py::array_t<float>, py::array_t<int>, int>(),
             py::arg("vocab_filename"),
             py::arg("map_desc_np"),
             py::arg("image_ids_np"),
             py::arg("n_images_total"))
        .def("match", &ImageRetrievalMatcher::match,
             py::arg("query_desc_np"),
             py::arg("ratio_threshold") = 0.80f,
             py::arg("top_k_images")    = 10);
}