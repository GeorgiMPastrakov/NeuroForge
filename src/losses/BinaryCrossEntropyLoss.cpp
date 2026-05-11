#include "neuroforge/losses/BinaryCrossEntropyLoss.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuroforge {

namespace {

constexpr double epsilon = 1e-12;

}

Tensor BinaryCrossEntropyLoss::forward(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("BinaryCrossEntropyLoss shape mismatch: prediction shape " + prediction.shape().toString() + " does not match target shape " + target.shape().toString() + ".");
    }

    prediction_ = prediction;
    target_ = target;

    if (prediction.requiresGrad()) {
        Tensor clipped = prediction.clamp(epsilon, 1.0 - epsilon);
        Tensor one = Tensor::ones(prediction.shape());
        Tensor positive = target.multiply(clipped.log());
        Tensor negative = one.subtract(target).multiply(one.subtract(clipped).log());
        return positive.add(negative).mean().multiply(-1.0);
    }

    double total = 0.0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        const double clipped = std::clamp(prediction.data()[index], epsilon, 1.0 - epsilon);
        const double target_value = target.data()[index];
        total -= target_value * std::log(clipped) + (1.0 - target_value) * std::log(1.0 - clipped);
    }

    return Tensor(std::vector<double>{total / static_cast<double>(prediction.size())}, Shape({1}));
}

Tensor BinaryCrossEntropyLoss::backward() {
    if (!prediction_.has_value() || !target_.has_value()) {
        throw std::invalid_argument("BinaryCrossEntropyLoss backward requires a prior forward call.");
    }

    Tensor result = Tensor::zeros(prediction_->shape());
    const double scale = 1.0 / static_cast<double>(prediction_->size());

    for (size_t index = 0; index < prediction_->size(); ++index) {
        const double clipped = std::clamp(prediction_->data()[index], epsilon, 1.0 - epsilon);
        const double target_value = target_->data()[index];
        result.data()[index] = (clipped - target_value) / (clipped * (1.0 - clipped)) * scale;
    }

    return result;
}

std::string BinaryCrossEntropyLoss::name() const {
    return "BinaryCrossEntropyLoss";
}

}
