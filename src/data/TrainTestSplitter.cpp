#include "neuroforge/data/TrainTestSplitter.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace neuroforge {

TrainTestSplit TrainTestSplitter::split(const Dataset& dataset, double train_ratio, bool shuffle, uint32_t seed) {
    if (dataset.size() < 2) {
        throw std::invalid_argument("TrainTestSplitter requires at least two samples.");
    }

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        throw std::invalid_argument("TrainTestSplitter train ratio must be between 0 and 1.");
    }

    std::vector<size_t> indices(dataset.size());
    std::iota(indices.begin(), indices.end(), size_t{0});

    if (shuffle) {
        std::mt19937 generator(seed);
        std::shuffle(indices.begin(), indices.end(), generator);
    }

    size_t train_size = static_cast<size_t>(static_cast<double>(dataset.size()) * train_ratio);

    if (train_size == 0) {
        train_size = 1;
    }

    if (train_size >= dataset.size()) {
        train_size = dataset.size() - 1;
    }

    std::vector<size_t> train_indices(indices.begin(), indices.begin() + static_cast<std::ptrdiff_t>(train_size));
    std::vector<size_t> test_indices(indices.begin() + static_cast<std::ptrdiff_t>(train_size), indices.end());

    return {dataset.subset(train_indices), dataset.subset(test_indices)};
}

}
