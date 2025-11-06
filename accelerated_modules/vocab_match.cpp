#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <limits>

namespace py = pybind11;

class VocabMatcher {
private:
    std::vector<std::vector<float>> vocabulary;
    std::vector<std::vector<float>> map_descriptors;
    std::vector<std::vector<int>> word_groups;  // word_id -> list of map indices
    int top_k;
    int desc_dim;
    
    float compute_distance(const std::vector<float>& a, const std::vector<float>& b) {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
    
    int find_nearest_word(const std::vector<float>& descriptor) {
        float min_dist = std::numeric_limits<float>::max();
        int best_word = 0;
        
        for (size_t i = 0; i < vocabulary.size(); i++) {
            float dist = compute_distance(vocabulary[i], descriptor);
            if (dist < min_dist) {
                min_dist = dist;
                best_word = i;
            }
        }
        return best_word;
    }
    
public:
    VocabMatcher(py::array_t<float> vocab_np, 
                 py::array_t<float> map_desc_np,
                 int k = 3) : top_k(k) {
        
        auto vocab_buf = vocab_np.request();
        auto map_buf = map_desc_np.request();
        
        int n_words = vocab_buf.shape[0];
        int n_map = map_buf.shape[0];
        desc_dim = vocab_buf.shape[1];
        
        // Copy vocabulary
        vocabulary.resize(n_words);
        float* vocab_ptr = (float*)vocab_buf.ptr;
        for (int i = 0; i < n_words; i++) {
            vocabulary[i].resize(desc_dim);
            for (int j = 0; j < desc_dim; j++) {
                vocabulary[i][j] = vocab_ptr[i * desc_dim + j];
            }
        }
        
        // Copy map descriptors
        map_descriptors.resize(n_map);
        float* map_ptr = (float*)map_buf.ptr;
        for (int i = 0; i < n_map; i++) {
            map_descriptors[i].resize(desc_dim);
            for (int j = 0; j < desc_dim; j++) {
                map_descriptors[i][j] = map_ptr[i * desc_dim + j];
            }
        }
        
        // Build inverted index
        word_groups.resize(n_words);
        for (int i = 0; i < n_map; i++) {
            int word_id = find_nearest_word(map_descriptors[i]);
            word_groups[word_id].push_back(i);
        }
    }
    
    py::tuple match(py::array_t<float> query_desc_np, float ratio_threshold = 0.80) {
        auto query_buf = query_desc_np.request();
        int n_query = query_buf.shape[0];
        float* query_ptr = (float*)query_buf.ptr;
        
        std::vector<int> match_query_idx;
        std::vector<int> match_map_idx;
        std::vector<float> match_distances;
        
        for (int query_idx = 0; query_idx < n_query; query_idx++) {
            // Get query descriptor
            std::vector<float> query_desc(desc_dim);
            for (int j = 0; j < desc_dim; j++) {
                query_desc[j] = query_ptr[query_idx * desc_dim + j];
            }
            
            // Find top-k nearest words
            std::vector<std::pair<float, int>> word_distances;
            for (size_t i = 0; i < vocabulary.size(); i++) {
                float dist = compute_distance(vocabulary[i], query_desc);
                word_distances.push_back({dist, i});
            }
            std::partial_sort(word_distances.begin(), 
                            word_distances.begin() + std::min(top_k, (int)word_distances.size()),
                            word_distances.end());
            
            // Collect candidates from top-k words - REMOVE DUPLICATES
            std::vector<int> candidates;
            std::unordered_set<int> seen;
            for (int i = 0; i < std::min(top_k, (int)word_distances.size()); i++) {
                int word_id = word_distances[i].second;
                for (int map_idx : word_groups[word_id]) {
                    if (seen.find(map_idx) == seen.end()) {
                        candidates.push_back(map_idx);
                        seen.insert(map_idx);
                    }
                }
            }
            
            if (candidates.size() < 2) continue;
            
            // Find best and second-best among candidates
            float best_dist = std::numeric_limits<float>::max();
            float second_best_dist = std::numeric_limits<float>::max();
            int best_idx = -1;
            
            for (int map_idx : candidates) {
                float dist = compute_distance(query_desc, map_descriptors[map_idx]);
                
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
        
        // Convert to numpy arrays
        auto query_idx_out = py::array_t<int>(match_query_idx.size());
        auto map_idx_out = py::array_t<int>(match_map_idx.size());
        auto distances_out = py::array_t<float>(match_distances.size());
        
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

PYBIND11_MODULE(vocab_match, m) {
    py::class_<VocabMatcher>(m, "VocabMatcher")
        .def(py::init<py::array_t<float>, py::array_t<float>, int>(),
             py::arg("vocabulary"),
             py::arg("map_descriptors"),
             py::arg("top_k") = 3)
        .def("match", &VocabMatcher::match,
             py::arg("query_descriptors"),
             py::arg("ratio_threshold") = 0.80);
}

