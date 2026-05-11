#include "neuroforge/nn/Softmax.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor Softmax::forward(const Tensor& input) {
    cached_output_ = input.softmaxRows();
    return cached_output_.value();
}

Tensor Softmax::backward(const Tensor& grad_output) {
    if (!cached_output_.has_value()) {
        throw std::invalid_argument("Softmax backward requires a prior forward call.");
    }

    if (grad_output.shape() != cached_output_->shape()) {
        throw std::invalid_argument("Softmax grad output shape mismatch: expected " + cached_output_->shape().toString() + ", got " + grad_output.shape().toString() + ".");
    }

    if (grad_output.rank() != 2) {
        throw std::invalid_argument("Softmax backward requires rank 2 grad output.");
    }

    const size_t rows = grad_output.shape().rows();
    const size_t cols = grad_output.shape().cols();
    Tensor result = Tensor::zeros(grad_output.shape());

    for (size_t row = 0; row < rows; ++row) {
        double weighted_gradient = 0.0;

        for (size_t col = 0; col < cols; ++col) {
            weighted_gradient += grad_output.at(row, col) * cached_output_->at(row, col);
        }

        for (size_t col = 0; col < cols; ++col) {
            result.at(row, col) = cached_output_->at(row, col) * (grad_output.at(row, col) - weighted_gradient);
        }
    }

    return result;
}

std::string Softmax::name() const {
    return "Softmax()";
}

}
