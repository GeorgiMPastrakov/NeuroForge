#include "neuroforge/nn/Tanh.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor Tanh::forward(const Tensor& input) {
    cached_output_ = input.tanh();
    return cached_output_.value();
}

Tensor Tanh::backward(const Tensor& grad_output) {
    if (!cached_output_.has_value()) {
        throw std::invalid_argument("Tanh backward requires a prior forward call.");
    }

    if (grad_output.shape() != cached_output_->shape()) {
        throw std::invalid_argument("Tanh grad output shape mismatch: expected " + cached_output_->shape().toString() + ", got " + grad_output.shape().toString() + ".");
    }

    Tensor result = Tensor::zeros(grad_output.shape());

    for (size_t index = 0; index < grad_output.size(); ++index) {
        const double value = cached_output_->data()[index];
        result.data()[index] = grad_output.data()[index] * (1.0 - value * value);
    }

    return result;
}

std::string Tanh::name() const {
    return "Tanh()";
}

}
