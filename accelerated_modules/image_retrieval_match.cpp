#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
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

    std::vector<std::vector<int>> word_to_points;
    std::vector<std::vector<int>> word_to_images;

    std::vector<std::vector<float>> map_descriptors;
    std::vector<int> map_image_ids;
    int n_images;

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

    // Collect candidate map point indices from top-K images via voting
    std::vector<int> get_candidates(const std::vector<std::vector<float>>& query_descs,
                                    int top_k_images) {
        std::vector<int> image_votes(n_images, 0);
        for (const auto& qdesc : query_descs) {
            int word_id = query_tree(qdesc);
            for (int img_idx : word_to_images[word_id])
                image_votes[img_idx]++;
        }

        int k = std::min(top_k_images, n_images);
        std::vector<int> image_order(n_images);
        for (int i = 0; i < n_images; i++) image_order[i] = i;
        std::partial_sort(image_order.begin(), image_order.begin() + k, image_order.end(),
                          [&](int a, int b){ return image_votes[a] > image_votes[b]; });

        std::unordered_set<int> top_image_set(image_order.begin(), image_order.begin() + k);

        std::vector<int> candidates;
        candidates.reserve(query_descs.size() * 50);
        for (int i = 0; i < (int)map_descriptors.size(); i++) {
            if (top_image_set.count(map_image_ids[i]))
                candidates.push_back(i);
        }
        return candidates;
    }

    // Compute squared L2 distance matrix using Eigen GEMM:
    // D[i,j] = ||q_i - c_j||^2 = ||q_i||^2 + ||c_j||^2 - 2 * q_i . c_j
    // Returns matrix of shape (n_query x n_candidates), squared distances
    Eigen::MatrixXf compute_distance_matrix(
            const std::vector<std::vector<float>>& query_descs,
            const std::vector<int>& candidates) {

        int n_query = (int)query_descs.size();
        int n_cands = (int)candidates.size();

        // Build query matrix (n_query x dim)
        Eigen::MatrixXf Q(n_query, dim);
        for (int i = 0; i < n_query; i++)
            for (int j = 0; j < dim; j++)
                Q(i, j) = query_descs[i][j];

        // Build candidate matrix (n_cands x dim)
        Eigen::MatrixXf C(n_cands, dim);
        for (int i = 0; i < n_cands; i++)
            for (int j = 0; j < dim; j++)
                C(i, j) = map_descriptors[candidates[i]][j];

        // Squared norms
        Eigen::VectorXf q_norms = Q.rowwise().squaredNorm();  // (n_query)
        Eigen::VectorXf c_norms = C.rowwise().squaredNorm();  // (n_cands)

        // D = q_norms + c_norms^T - 2 * Q @ C^T
        Eigen::MatrixXf D = -2.0f * (Q * C.transpose());
        D.colwise() += q_norms;
        D.rowwise() += c_norms.transpose();
        D = D.cwiseMax(0.0f);  // Clamp numerical negatives to zero

        return D;  // Squared distances — take sqrt only when needed
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

    // Standard match with ratio test
    py::tuple match(py::array_t<float> query_desc_np,
                    float ratio_threshold = 0.80f,
                    int top_k_images = 10) {

        auto query_buf = query_desc_np.request();
        int n_query = query_buf.shape[0];
        float* query_ptr = (float*)query_buf.ptr;

        std::vector<std::vector<float>> query_descs(n_query, std::vector<float>(dim));
        for (int qi = 0; qi < n_query; qi++)
            for (int j = 0; j < dim; j++)
                query_descs[qi][j] = query_ptr[qi * dim + j];

        std::vector<int> candidates = get_candidates(query_descs, top_k_images);
        if (candidates.size() < 2) return py::make_tuple(
            py::array_t<int>(0), py::array_t<int>(0), py::array_t<float>(0));

        Eigen::MatrixXf D = compute_distance_matrix(query_descs, candidates);

        std::vector<int>   match_query_idx;
        std::vector<int>   match_map_idx;
        std::vector<float> match_distances;

        float ratio_sq = ratio_threshold * ratio_threshold;

        for (int qi = 0; qi < n_query; qi++) {
            float best_sq        = std::numeric_limits<float>::max();
            float second_best_sq = std::numeric_limits<float>::max();
            int   best_idx       = -1;

            for (int ci = 0; ci < (int)candidates.size(); ci++) {
                float d = D(qi, ci);
                if (d < best_sq) {
                    second_best_sq = best_sq;
                    best_sq = d;
                    best_idx = ci;
                } else if (d < second_best_sq) {
                    second_best_sq = d;
                }
            }

            if (best_idx >= 0 && best_sq < ratio_sq * second_best_sq) {
                match_query_idx.push_back(qi);
                match_map_idx.push_back(candidates[best_idx]);
                match_distances.push_back(std::sqrt(best_sq));
            }
        }

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

    // Returns ALL candidate matches with ranks — no ratio filtering
    py::tuple match_with_stats(py::array_t<float> query_desc_np,
                               int top_k_images = 10) {

        auto query_buf = query_desc_np.request();
        int n_query = query_buf.shape[0];
        float* query_ptr = (float*)query_buf.ptr;

        std::vector<std::vector<float>> query_descs(n_query, std::vector<float>(dim));
        for (int qi = 0; qi < n_query; qi++)
            for (int j = 0; j < dim; j++)
                query_descs[qi][j] = query_ptr[qi * dim + j];

        std::vector<int> candidates = get_candidates(query_descs, top_k_images);
        if (candidates.empty()) return py::make_tuple(
            py::array_t<int>(0), py::array_t<int>(0),
            py::array_t<float>(0), py::array_t<int>(0));

        Eigen::MatrixXf D = compute_distance_matrix(query_descs, candidates);

        std::vector<int>   match_query_idx;
        std::vector<int>   match_map_idx;
        std::vector<float> match_distances;
        std::vector<int>   match_ranks;

        for (int qi = 0; qi < n_query; qi++) {
            // Collect and sort distances for this query
            std::vector<std::pair<float, int>> dist_idx(candidates.size());
            for (int ci = 0; ci < (int)candidates.size(); ci++)
                dist_idx[ci] = {D(qi, ci), ci};

            std::sort(dist_idx.begin(), dist_idx.end());

            for (int rank = 0; rank < (int)dist_idx.size(); rank++) {
                match_query_idx.push_back(qi);
                match_map_idx.push_back(candidates[dist_idx[rank].second]);
                match_distances.push_back(std::sqrt(dist_idx[rank].first));
                match_ranks.push_back(rank);
            }
        }

        py::array_t<int>   query_idx_out(match_query_idx.size());
        py::array_t<int>   map_idx_out(match_map_idx.size());
        py::array_t<float> distances_out(match_distances.size());
        py::array_t<int>   ranks_out(match_ranks.size());

        auto q_ptr = query_idx_out.mutable_unchecked<1>();
        auto m_ptr = map_idx_out.mutable_unchecked<1>();
        auto d_ptr = distances_out.mutable_unchecked<1>();
        auto r_ptr = ranks_out.mutable_unchecked<1>();

        for (size_t i = 0; i < match_query_idx.size(); i++) {
            q_ptr(i) = match_query_idx[i];
            m_ptr(i) = match_map_idx[i];
            d_ptr(i) = match_distances[i];
            r_ptr(i) = match_ranks[i];
        }

        return py::make_tuple(query_idx_out, map_idx_out, distances_out, ranks_out);
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
             py::arg("top_k_images")    = 10)
        .def("match_with_stats", &ImageRetrievalMatcher::match_with_stats,
             py::arg("query_desc_np"),
             py::arg("top_k_images")    = 10);
}