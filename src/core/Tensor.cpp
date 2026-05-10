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

    const bool requires_grad = requiresGrad() || other.requiresGrad();
    Tensor output(std::move(result), node_->shape, requires_grad);

    if (requires_grad) {
        auto left = node_;
        auto right = other.node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {left, right};
        output_node->backward = [weak_output, left, right] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                if (left->requires_grad) {
                    left->backward_grad[index] += current->backward_grad[index];
                }

                if (right->requires_grad) {
                    right->backward_grad[index] += current->backward_grad[index];
                }
            }
        };
    }

    return output;
}

Tensor Tensor::subtract(const Tensor& other) const {
    requireSameShape(other, "Subtract");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(node_->data[index] - other.node_->data[index]);
    }

    const bool requires_grad = requiresGrad() || other.requiresGrad();
    Tensor output(std::move(result), node_->shape, requires_grad);

    if (requires_grad) {
        auto left = node_;
        auto right = other.node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {left, right};
        output_node->backward = [weak_output, left, right] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                if (left->requires_grad) {
                    left->backward_grad[index] += current->backward_grad[index];
                }

                if (right->requires_grad) {
                    right->backward_grad[index] -= current->backward_grad[index];
                }
            }
        };
    }

    return output;
}

Tensor Tensor::multiply(const Tensor& other) const {
    requireSameShape(other, "Multiply");

    std::vector<double> result;
    result.reserve(size());

    for (size_t index = 0; index < size(); ++index) {
        result.push_back(node_->data[index] * other.node_->data[index]);
    }

    const bool requires_grad = requiresGrad() || other.requiresGrad();
    Tensor output(std::move(result), node_->shape, requires_grad);

    if (requires_grad) {
        auto left = node_;
        auto right = other.node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {left, right};
        output_node->backward = [weak_output, left, right] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                if (left->requires_grad) {
                    left->backward_grad[index] += current->backward_grad[index] * right->data[index];
                }

                if (right->requires_grad) {
                    right->backward_grad[index] += current->backward_grad[index] * left->data[index];
                }
            }
        };
    }

    return output;
}

Tensor Tensor::multiply(double scalar) const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(value * scalar);
    }

    Tensor output(std::move(result), node_->shape, requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent, scalar] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                parent->backward_grad[index] += current->backward_grad[index] * scalar;
            }
        };
    }

    return output;
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

    const bool requires_grad = requiresGrad() || other.requiresGrad();
    Tensor output(std::move(result), Shape({rows, cols}), requires_grad);

    if (requires_grad) {
        auto left = node_;
        auto right = other.node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {left, right};
        output_node->backward = [weak_output, left, right, rows, inner, cols] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            if (left->requires_grad) {
                for (size_t row = 0; row < rows; ++row) {
                    for (size_t index = 0; index < inner; ++index) {
                        double total = 0.0;

                        for (size_t col = 0; col < cols; ++col) {
                            total += current->backward_grad[row * cols + col] * right->data[index * cols + col];
                        }

                        left->backward_grad[row * inner + index] += total;
                    }
                }
            }

            if (right->requires_grad) {
                for (size_t index = 0; index < inner; ++index) {
                    for (size_t col = 0; col < cols; ++col) {
                        double total = 0.0;

                        for (size_t row = 0; row < rows; ++row) {
                            total += left->data[row * inner + index] * current->backward_grad[row * cols + col];
                        }

                        right->backward_grad[index * cols + col] += total;
                    }
                }
            }
        };
    }

    return output;
}

Tensor Tensor::addRowVector(const Tensor& row_vector) const {
    if (rank() != 2 || row_vector.rank() != 2) {
        throw std::invalid_argument("AddRowVector requires rank 2 tensors.");
    }

    if (row_vector.shape().rows() != 1 || row_vector.shape().cols() != shape().cols()) {
        throw std::invalid_argument("AddRowVector shape mismatch: left shape " + shape().toString() + " cannot add row vector shape " + row_vector.shape().toString() + ".");
    }

    const size_t rows = shape().rows();
    const size_t cols = shape().cols();
    std::vector<double> result;
    result.reserve(size());

    for (size_t row = 0; row < rows; ++row) {
        for (size_t col = 0; col < cols; ++col) {
            result.push_back(at(row, col) + row_vector.at(0, col));
        }
    }

    const bool requires_grad = requiresGrad() || row_vector.requiresGrad();
    Tensor output(std::move(result), shape(), requires_grad);

    if (requires_grad) {
        auto left = node_;
        auto right = row_vector.node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {left, right};
        output_node->backward = [weak_output, left, right, rows, cols] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            if (left->requires_grad) {
                for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                    left->backward_grad[index] += current->backward_grad[index];
                }
            }

            if (right->requires_grad) {
                for (size_t col = 0; col < cols; ++col) {
                    double total = 0.0;

                    for (size_t row = 0; row < rows; ++row) {
                        total += current->backward_grad[row * cols + col];
                    }

                    right->backward_grad[col] += total;
                }
            }
        };
    }

    return output;
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

Tensor Tensor::pow(double exponent) const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(std::pow(value, exponent));
    }

    Tensor output(std::move(result), node_->shape, requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent, exponent] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                parent->backward_grad[index] += current->backward_grad[index] * exponent * std::pow(parent->data[index], exponent - 1.0);
            }
        };
    }

    return output;
}

Tensor Tensor::relu() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(value > 0.0 ? value : 0.0);
    }

    Tensor output(std::move(result), node_->shape, requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                parent->backward_grad[index] += parent->data[index] > 0.0 ? current->backward_grad[index] : 0.0;
            }
        };
    }

    return output;
}

Tensor Tensor::sigmoid() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(1.0 / (1.0 + std::exp(-value)));
    }

    Tensor output(std::move(result), node_->shape, requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                const double output_value = current->data[index];
                parent->backward_grad[index] += current->backward_grad[index] * output_value * (1.0 - output_value);
            }
        };
    }

    return output;
}

Tensor Tensor::tanh() const {
    std::vector<double> result;
    result.reserve(size());

    for (double value : node_->data) {
        result.push_back(std::tanh(value));
    }

    Tensor output(std::move(result), node_->shape, requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < current->backward_grad.size(); ++index) {
                const double output_value = current->data[index];
                parent->backward_grad[index] += current->backward_grad[index] * (1.0 - output_value * output_value);
            }
        };
    }

    return output;
}

Tensor Tensor::sum() const {
    double total = 0.0;

    for (double value : node_->data) {
        total += value;
    }

    Tensor output({total}, Shape({1}), requiresGrad());

    if (requiresGrad()) {
        auto parent = node_;
        auto output_node = output.node_;
        std::weak_ptr<Node> weak_output = output_node;
        output_node->parents = {parent};
        output_node->backward = [weak_output, parent] {
            auto current = weak_output.lock();
            if (!current) {
                return;
            }

            for (size_t index = 0; index < parent->backward_grad.size(); ++index) {
                parent->backward_grad[index] += current->backward_grad[0];
            }
        };
    }

    return output;
}

Tensor Tensor::mean() const {
    return sum().multiply(1.0 / static_cast<double>(size()));
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
