#include "neuroforge/nn/ReLU.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor ReLU::forward(const Tensor& input) {
    cached_input_ = input;
    return input.relu();
}

Tensor ReLU::backward(const Tensor& grad_output) {
    if (!cached_input_.has_value()) {
        throw std::invalid_argument("ReLU backward requires a prior forward call.");
    }

    if (grad_output.shape() != cached_input_->shape()) {
        throw std::invalid_argument("ReLU grad output shape mismatch: expected " + cached_input_->shape().toString() + ", got " + grad_output.shape().toString() + ".");
    }

    Tensor result = Tensor::zeros(grad_output.shape());

    for (size_t index = 0; index < grad_output.size(); ++index) {
        result.data()[index] = cached_input_->data()[index] > 0.0 ? grad_output.data()[index] : 0.0;
    }

    return result;
}

std::string ReLU::name() const {
    return "ReLU()";
}

}
