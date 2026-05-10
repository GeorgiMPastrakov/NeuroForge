#include "neuroforge/losses/MSELoss.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor MSELoss::forward(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("MSELoss shape mismatch: prediction shape " + prediction.shape().toString() + " does not match target shape " + target.shape().toString() + ".");
    }

    prediction_ = prediction;
    target_ = target;

    double total = 0.0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        const double diff = prediction.data()[index] - target.data()[index];
        total += diff * diff;
    }

    return Tensor(std::vector<double>{total / static_cast<double>(prediction.size())}, Shape({1}));
}

Tensor MSELoss::backward() {
    if (!prediction_.has_value() || !target_.has_value()) {
        throw std::invalid_argument("MSELoss backward requires a prior forward call.");
    }

    Tensor result = Tensor::zeros(prediction_->shape());
    const double scale = 2.0 / static_cast<double>(prediction_->size());

    for (size_t index = 0; index < prediction_->size(); ++index) {
        result.data()[index] = scale * (prediction_->data()[index] - target_->data()[index]);
    }

    return result;
}

std::string MSELoss::name() const {
    return "MSELoss";
}

}
