#include "neuroforge/nn/Linear.hpp"

#include "neuroforge/core/Random.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace neuroforge {

Linear::Linear(size_t in_features, size_t out_features)
    : in_features_(in_features),
      out_features_(out_features),
      weights_(Random::uniform(Shape({in_features, out_features}), -1.0 / std::sqrt(static_cast<double>(in_features)), 1.0 / std::sqrt(static_cast<double>(in_features))), "weights"),
      bias_(Tensor::zeros(Shape({1, out_features})), "bias") {
    if (in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Linear features must be greater than zero.");
    }
}

Tensor Linear::forward(const Tensor& input) {
    validateInput(input);
    cached_input_ = input;

    return input.matmul(weights_.data()).addRowVector(bias_.data());
}

Tensor Linear::backward(const Tensor& grad_output) {
    if (!cached_input_.has_value()) {
        throw std::invalid_argument("Linear backward requires a prior forward call.");
    }

    validateGradOutput(grad_output);

    const Tensor& input = cached_input_.value();
    Tensor weight_grad = input.transpose().matmul(grad_output);
    Tensor input_grad = grad_output.matmul(weights_.data().transpose());

    for (size_t index = 0; index < weights_.grad().size(); ++index) {
        weights_.grad().data()[index] += weight_grad.data()[index];
    }

    for (size_t col = 0; col < out_features_; ++col) {
        double total = 0.0;

        for (size_t row = 0; row < grad_output.shape().rows(); ++row) {
            total += grad_output.at(row, col);
        }

        bias_.grad().at(0, col) += total;
    }

    return input_grad;
}

std::vector<Parameter*> Linear::parameters() {
    return {&weights_, &bias_};
}

std::string Linear::name() const {
    return "Linear(" + std::to_string(in_features_) + " -> " + std::to_string(out_features_) + ")";
}

Parameter& Linear::weights() {
    return weights_;
}

const Parameter& Linear::weights() const {
    return weights_;
}

Parameter& Linear::bias() {
    return bias_;
}

const Parameter& Linear::bias() const {
    return bias_;
}

size_t Linear::inFeatures() const {
    return in_features_;
}

size_t Linear::outFeatures() const {
    return out_features_;
}

size_t Linear::parameterCount() const {
    return in_features_ * out_features_ + out_features_;
}

void Linear::validateInput(const Tensor& input) const {
    if (input.rank() != 2) {
        throw std::invalid_argument("Linear input must have rank 2.");
    }

    if (input.shape().cols() != in_features_) {
        throw std::invalid_argument("Linear input shape mismatch: expected [batch, " + std::to_string(in_features_) + "], got " + input.shape().toString() + ".");
    }
}

void Linear::validateGradOutput(const Tensor& grad_output) const {
    if (grad_output.rank() != 2) {
        throw std::invalid_argument("Linear grad output must have rank 2.");
    }

    if (grad_output.shape().cols() != out_features_) {
        throw std::invalid_argument("Linear grad output shape mismatch: expected [batch, " + std::to_string(out_features_) + "], got " + grad_output.shape().toString() + ".");
    }

    if (cached_input_.has_value() && grad_output.shape().rows() != cached_input_->shape().rows()) {
        throw std::invalid_argument("Linear grad output batch size must match cached input batch size.");
    }
}

}
