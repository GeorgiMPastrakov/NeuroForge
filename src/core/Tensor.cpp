#include "neuroforge/core/Tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace neuroforge {

Tensor::Node::Node(std::vector<double> data_value, Shape shape_value, bool requires_grad_value)
    : data(std::move(data_value)),
      shape(std::move(shape_value)),
      grad(data.size(), 0.0),
      backward_grad(data.size(), 0.0),
      requires_grad(requires_grad_value) {
}

Tensor::Tensor(std::vector<double> data, Shape shape, bool requires_grad)
    : node_(nullptr) {
    if (data.size() != shape.size()) {
        throw std::invalid_argument("Tensor data size does not match shape " + shape.toString() + ".");
    }

    node_ = std::make_shared<Node>(std::move(data), std::move(shape), requires_grad);
}

Tensor::Tensor(std::shared_ptr<Node> node)
    : node_(std::move(node)) {
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
    return node_->shape;
}

size_t Tensor::rank() const {
    return node_->shape.rank();
}

size_t Tensor::size() const {
    return node_->data.size();
}

const std::vector<double>& Tensor::data() const {
    return node_->data;
}

std::vector<double>& Tensor::data() {
    return node_->data;
}

double Tensor::at(size_t index) const {
    if (rank() != 1) {
        throw std::invalid_argument("Tensor 1D indexing requires rank 1 tensor.");
    }

    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return node_->data[index];
}

double& Tensor::at(size_t index) {
    if (rank() != 1) {
        throw std::invalid_argument("Tensor 1D indexing requires rank 1 tensor.");
    }

    if (index >= size()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return node_->data[index];
}

double Tensor::at(size_t row, size_t col) const {
    return node_->data[flatIndex(row, col)];
}

double& Tensor::at(size_t row, size_t col) {
    return node_->data[flatIndex(row, col)];
}

Tensor Tensor::add(const Tensor& other) const {
    requireSameShape(other, "Add");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(node_->data[index] + other.node_->data[index]);
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::subtract(const Tensor& other) const {
    requireSameShape(other, "Subtract");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(node_->data[index] - other.node_->data[index]);
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::multiply(const Tensor& other) const {
    requireSameShape(other, "Multiply");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(node_->data[index] * other.node_->data[index]);
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::multiply(double scalar) const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(value * scalar);
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::matmul(const Tensor& other) const {
    if (rank() != 2 || other.rank() != 2) {
        throw std::invalid_argument("MatMul requires rank 2 tensors.");
    }

    if (node_->shape.cols() != other.node_->shape.rows()) {
        throw std::invalid_argument("MatMul shape mismatch: left shape " + node_->shape.toString() + " cannot multiply right shape " + other.node_->shape.toString() + ".");
    }

    const size_t rows = node_->shape.rows();
    const size_t inner = node_->shape.cols();
    const size_t cols = other.node_->shape.cols();
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

    for (size_t row = 0; row < node_->shape.rows(); ++row) {
        for (size_t col = 0; col < node_->shape.cols(); ++col) {
            result[col * node_->shape.rows() + row] = at(row, col);
        }
    }

    return Tensor(std::move(result), Shape({node_->shape.cols(), node_->shape.rows()}));
}

Tensor Tensor::relu() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(value > 0.0 ? value : 0.0);
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::sigmoid() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(1.0 / (1.0 + std::exp(-value)));
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::tanh() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(std::tanh(value));
    }

    return Tensor(std::move(result), node_->shape);
}

Tensor Tensor::sum() const {
    double total = 0.0;

    for (double value : node_->data) {
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

    return node_->data[0];
}

bool Tensor::requiresGrad() const {
    return node_->requires_grad;
}

void Tensor::setRequiresGrad(bool requires_grad) {
    node_->requires_grad = requires_grad;
}

const std::vector<double>& Tensor::grad() const {
    return node_->grad;
}

std::vector<double>& Tensor::grad() {
    return node_->grad;
}

void Tensor::zeroGrad() {
    node_->grad.assign(size(), 0.0);
    node_->backward_grad.assign(size(), 0.0);
}

void Tensor::backward() {
    if (size() != 1) {
        throw std::invalid_argument("Tensor backward requires a scalar tensor.");
    }

    std::unordered_set<Node*> visited;
    std::vector<std::shared_ptr<Node>> order;

    auto build = [&](const std::shared_ptr<Node>& start, auto&& build_ref) -> void {
        if (visited.count(start.get()) > 0) {
            return;
        }

        visited.insert(start.get());

        for (const auto& parent : start->parents) {
            build_ref(parent, build_ref);
        }

        order.push_back(start);
    };

    build(node_, build);

    for (const auto& node : order) {
        node->backward_grad.assign(node->data.size(), 0.0);
    }

    node_->backward_grad[0] = 1.0;

    for (auto iterator = order.rbegin(); iterator != order.rend(); ++iterator) {
        if ((*iterator)->backward) {
            (*iterator)->backward();
        }
    }

    for (const auto& node : order) {
        for (size_t index = 0; index < node->grad.size(); ++index) {
            node->grad[index] += node->backward_grad[index];
        }
    }
}

size_t Tensor::flatIndex(size_t row, size_t col) const {
    if (rank() != 2) {
        throw std::invalid_argument("Tensor 2D indexing requires rank 2 tensor.");
    }

    if (row >= node_->shape.rows() || col >= node_->shape.cols()) {
        throw std::out_of_range("Tensor index out of range.");
    }

    return row * node_->shape.cols() + col;
}

void Tensor::requireSameShape(const Tensor& other, const std::string& operation) const {
    if (node_->shape != other.node_->shape) {
        throw std::invalid_argument(operation + " shape mismatch: left shape " + node_->shape.toString() + " does not match right shape " + other.node_->shape.toString() + ".");
    }
}

}
