#include "neuroforge/data/Dataset.hpp"

#include <stdexcept>
#include <utility>

namespace neuroforge {

Dataset::Dataset(Tensor features, Tensor labels)
    : features_(std::move(features)), labels_(std::move(labels)) {
    if (features_.rank() != 2 || labels_.rank() != 2) {
        throw std::invalid_argument("Dataset features and labels must be rank 2 tensors.");
    }

    if (features_.shape().rows() != labels_.shape().rows()) {
        throw std::invalid_argument("Dataset feature rows must match label rows.");
    }
}

size_t Dataset::size() const {
    return features_.shape().rows();
}

const Tensor& Dataset::features() const {
    return features_;
}

const Tensor& Dataset::labels() const {
    return labels_;
}

Batch Dataset::sample(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("Dataset sample index out of range.");
    }

    std::vector<double> feature_values;
    feature_values.reserve(features_.shape().cols());

    for (size_t col = 0; col < features_.shape().cols(); ++col) {
        feature_values.push_back(features_.at(index, col));
    }

    std::vector<double> label_values;
    label_values.reserve(labels_.shape().cols());

    for (size_t col = 0; col < labels_.shape().cols(); ++col) {
        label_values.push_back(labels_.at(index, col));
    }

    return {
        Tensor(std::move(feature_values), Shape({1, features_.shape().cols()})),
        Tensor(std::move(label_values), Shape({1, labels_.shape().cols()}))
    };
}

Dataset Dataset::subset(const std::vector<size_t>& indices) const {
    if (indices.empty()) {
        throw std::invalid_argument("Dataset subset requires at least one index.");
    }

    std::vector<double> feature_values;
    std::vector<double> label_values;
    feature_values.reserve(indices.size() * features_.shape().cols());
    label_values.reserve(indices.size() * labels_.shape().cols());

    for (size_t index : indices) {
        if (index >= size()) {
            throw std::out_of_range("Dataset subset index out of range.");
        }

        for (size_t col = 0; col < features_.shape().cols(); ++col) {
            feature_values.push_back(features_.at(index, col));
        }

        for (size_t col = 0; col < labels_.shape().cols(); ++col) {
            label_values.push_back(labels_.at(index, col));
        }
    }

    return Dataset(
        Tensor(std::move(feature_values), Shape({indices.size(), features_.shape().cols()})),
        Tensor(std::move(label_values), Shape({indices.size(), labels_.shape().cols()}))
    );
}

}
