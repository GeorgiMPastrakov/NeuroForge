#include "neuroforge/nn/LeakyReLU.hpp"

#include <sstream>
#include <stdexcept>

namespace neuroforge {

LeakyReLU::LeakyReLU(double negative_slope)
    : negative_slope_(negative_slope) {
    if (negative_slope < 0.0) {
        throw std::invalid_argument("LeakyReLU negative slope must be nonnegative.");
    }
}

Tensor LeakyReLU::forward(const Tensor& input) {
    cached_input_ = input;
    return input.leakyRelu(negative_slope_);
}

Tensor LeakyReLU::backward(const Tensor& grad_output) {
    if (!cached_input_.has_value()) {
        throw std::invalid_argument("LeakyReLU backward requires a prior forward call.");
    }

    if (grad_output.shape() != cached_input_->shape()) {
        throw std::invalid_argument("LeakyReLU grad output shape mismatch: expected " + cached_input_->shape().toString() + ", got " + grad_output.shape().toString() + ".");
    }

    Tensor result = Tensor::zeros(grad_output.shape());

    for (size_t index = 0; index < grad_output.size(); ++index) {
        const double slope = cached_input_->data()[index] > 0.0 ? 1.0 : negative_slope_;
        result.data()[index] = grad_output.data()[index] * slope;
    }

    return result;
}

std::string LeakyReLU::name() const {
    std::ostringstream stream;
    stream << "LeakyReLU(" << negative_slope_ << ")";
    return stream.str();
}

}
