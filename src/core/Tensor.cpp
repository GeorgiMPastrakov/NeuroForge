#include "neuroforge/core/Tensor.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor::Tensor(std::vector<double> data, Shape shape)
    : data_(std::move(data)), shape_(std::move(shape)) {
    if (data_.size() != shape_.size()) {
        throw std::invalid_argument("Tensor data size does not match shape " + shape_.toString() + ".");
    }
}

Tensor Tensor::fromVector(std::initializer_list<std::initializer_list<double>> values) {
    std::vector<std::vector<double>> rows;
    rows.reserve(values.size());

    for (const auto& row : values) {
        rows.emplace_back(row);
    }

    return fromVector(rows);
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

Tensor Tensor::add(const Tensor& other) const {
    requireSameShape(other, "Add");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(data_[index] + other.data_[index]);
    }

    return Tensor(std::move(result), shape_);
}

Tensor Tensor::subtract(const Tensor& other) const {
    requireSameShape(other, "Subtract");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(data_[index] - other.data_[index]);
    }

    return Tensor(std::move(result), shape_);
}

Tensor Tensor::multiply(const Tensor& other) const {
    requireSameShape(other, "Multiply");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(data_[index] * other.data_[index]);
    }

    return Tensor(std::move(result), shape_);
}

Tensor Tensor::multiply(double scalar) const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : data_) {
        result.push_back(value * scalar);
    }

    return Tensor(std::move(result), shape_);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
        throw std::invalid_argument("MatMul requires rank 2 tensors.");
    }

    if (shape_.cols() != other.shape_.rows()) {
        throw std::invalid_argument("MatMul shape mismatch: left shape " + shape_.toString() + " cannot multiply right shape " + other.shape_.toString() + ".");
    }

    const size_t rows = shape_.rows();
    const size_t inner = shape_.cols();
    const size_t cols = other.shape_.cols();
    std::vector<double> result(rows * cols, 0.0);

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            double total = 0.0;

            for (size_t index = 0; index < inner; ++index) {
                total += at(row, index) * other.at(index, col);
            }

            result[row * cols + col] = total;
        }
    }

    return Tensor(std::move(result), Shape({rows, cols}));
}

Tensor Tensor::transpose() const {
    if (rank() != 2) {
        throw std::invalid_argument("Transpose requires rank 2 tensor.");
    }

    std::vector<double> result(size());

    for (size_t row = 0; row < shape_.rows(); ++row) {
        for (size_t col = 0; col < shape_.cols(); ++col) {
            result[col * shape_.rows() + row] = at(row, col);
        }
    }

    return Tensor(std::move(result), Shape({shape_.cols(), shape_.rows()}));
}

Tensor Tensor::sum() const {
    double total = 0.0;

    for (double value : data_) {
        total += value;
    }

    return Tensor({total}, Shape({1}));
}

Tensor Tensor::mean() const {
    return Tensor({sum().item() / static_cast<double>(size())}, Shape({1}));
}

double Tensor::item() const {
    if (size() != 1) {
        throw std::invalid_argument("Tensor item requires exactly one element.");
    }

    return data_[0];
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

void Tensor::requireSameShape(const Tensor& other, const std::string& operation) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument(operation + " shape mismatch: left shape " + shape_.toString() + " does not match right shape " + other.shape_.toString() + ".");
    }
}

}
