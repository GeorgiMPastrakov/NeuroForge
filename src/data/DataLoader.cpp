#include "neuroforge/data/DataLoader.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

namespace neuroforge {

DataLoader::DataLoader(const Dataset& dataset, size_t batch_size, bool shuffle, uint32_t seed)
    : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle), seed_(seed) {
    if (batch_size == 0) {
        throw std::invalid_argument("DataLoader batch size must be greater than zero.");
    }
}

std::vector<Batch> DataLoader::batches() const {
    std::vector<size_t> indices(dataset_.size());
    std::iota(indices.begin(), indices.end(), size_t{0});

    if (shuffle_) {
        std::mt19937 generator(seed_);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    std::vector<Batch> result;

    for (size_t start = 0; start < indices.size(); start += batch_size_) {
        const size_t end = std::min(start + batch_size_, indices.size());
        std::vector<size_t> batch_indices(indices.begin() + static_cast<std::ptrdiff_t>(start), indices.begin() + static_cast<std::ptrdiff_t>(end));
        Dataset batch = dataset_.subset(batch_indices);
        result.push_back({batch.features(), batch.labels()});
    }

    return result;
}

}
