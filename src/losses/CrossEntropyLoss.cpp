#include "neuroforge/losses/CrossEntropyLoss.hpp"

#include <cmath>
#include <stdexcept>

namespace neuroforge {

namespace {

constexpr double epsilon = 1e-12;

}

Tensor CrossEntropyLoss::forward(const Tensor& prediction, const Tensor& target) {
    if (prediction.rank() != 2 || target.rank() != 2) {
        throw std::invalid_argument("CrossEntropyLoss requires rank 2 prediction and target tensors.");
    }

    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("CrossEntropyLoss shape mismatch: prediction shape " + prediction.shape().toString() + " does not match target shape " + target.shape().toString() + ".");
    }

    Tensor probabilities = prediction.softmaxRows();
    probabilities_ = probabilities;
    target_ = target;

    if (prediction.requiresGrad()) {
        return probabilities.clamp(epsilon, 1.0).log().multiply(target).sum().multiply(-1.0 / static_cast<double>(prediction.shape().rows()));
    }

    double total = 0.0;

    for (size_t index = 0; index < probabilities.size(); ++index) {
        total -= target.data()[index] * std::log(probabilities.data()[index] < epsilon ? epsilon : probabilities.data()[index]);
    }

    return Tensor(std::vector<double>{total / static_cast<double>(prediction.shape().rows())}, Shape({1}));
}

Tensor CrossEntropyLoss::backward() {
    if (!probabilities_.has_value() || !target_.has_value()) {
        throw std::invalid_argument("CrossEntropyLoss backward requires a prior forward call.");
    }

    Tensor result = probabilities_->subtract(target_.value());
    return result.multiply(1.0 / static_cast<double>(probabilities_->shape().rows()));
}

std::string CrossEntropyLoss::name() const {
    return "CrossEntropyLoss";
}

}
