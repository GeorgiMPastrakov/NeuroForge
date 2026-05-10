#include "neuroforge/core/Tensor.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor::Tensor(std::vector<double> data, Shape shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
    if (data_.size() != shape_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape " + shape_.toString() + ".");
    }
}

Tensor Tensor::fromVector(const std::vector<double>& values) {
    return Tensor(values, Shape({values.size()}));
}

Tensor Tensor::fromVector(const std::vector<std::vector<double>>& values) {
    if (values.empty()) {
        throw std::invalid_argument("Tensor 2D input must not be empty.");
    }

    const size_t rows = values.size();
    const size_t cols = values.front().size();

    if (cols == 0) {
        throw std::invalid_argument("Tensor 2D rows must not be empty.");
    }

    std::vector<double> data;
    data.reserve(rows * cols);

    for (const auto& row : values) {
        if (row.size() != cols) {
            throw std::invalid_argument("Tensor 2D input rows must have the same length.");
        }

        data.insert(data.end(), row.begin(), row.end());
    }

    return Tensor(std::move(data), Shape({rows, cols}));
}

Tensor Tensor::zeros(const Shape& shape) {
    return full(shape, 0.0);
}

Tensor Tensor::ones(const Shape& shape) {
    return full(shape, 1.0);
}

Tensor Tensor::full(const Shape& shape, double value) {
    return Tensor(std::vector<double>(shape.size(), value), shape);
}

const Shape& Tensor::shape() const {
    return shape_;
}

size_t Tensor::rank() const {
    return shape_.rank();
}

size_t Tensor::size() const {
    return data_.size();
}

const std::vector<double>& Tensor::data() const {
    return data_;
}

std::vector<double>& Tensor::data() {
    return data_;
}

double Tensor::at(size_t index) const {
    if (rank() != 1) {
        throw std::invalid_argument("Tensor 1D indexing requires rank 1 tensor.");
    }

    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return data_[index];
}

double& Tensor::at(size_t index) {
    if (rank() != 1) {
        throw std::invalid_argument("Tensor 1D indexing requires rank 1 tensor.");
    }

    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return data_[index];
}

double Tensor::at(size_t row, size_t col) const {
    return data_[flatIndex(row, col)];
}

double& Tensor::at(size_t row, size_t col) {
    return data_[flatIndex(row, col)];
}

size_t Tensor::flatIndex(size_t row, size_t col) const {
    if (rank() != 2) {
        throw std::invalid_argument("Tensor 2D indexing requires rank 2 tensor.");
    }

    if (row >= shape_.rows() || col >= shape_.cols()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return row * shape_.cols() + col;
}

}
