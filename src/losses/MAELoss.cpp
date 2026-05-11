#include "neuroforge/losses/MAELoss.hpp"

#include <stdexcept>

namespace neuroforge {

Tensor MAELoss::forward(const Tensor& prediction, const Tensor& target) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument("MAELoss shape mismatch: prediction shape " + prediction.shape().toString() + " does not match target shape " + target.shape().toString() + ".");
    }

    prediction_ = prediction;
    target_ = target;

    if (prediction.requiresGrad()) {
        return prediction.subtract(target).abs().mean();
    }

    double total = 0.0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        const double diff = prediction.data()[index] - target.data()[index];
        total += diff < 0.0 ? -diff : diff;
    }

    return Tensor(std::vector<double>{total / static_cast<double>(prediction.size())}, Shape({1}));
}

Tensor MAELoss::backward() {
    if (!prediction_.has_value() || !target_.has_value()) {
        throw std::invalid_argument("MAELoss backward requires a prior forward call.");
    }

    Tensor result = Tensor::zeros(prediction_->shape());
    const double scale = 1.0 / static_cast<double>(prediction_->size());

    for (size_t index = 0; index < prediction_->size(); ++index) {
        const double diff = prediction_->data()[index] - target_->data()[index];

        if (diff > 0.0) {
            result.data()[index] = scale;
        } else if (diff < 0.0) {
            result.data()[index] = -scale;
        }
    }

    return result;
}

std::string MAELoss::name() const {
    return "MAELoss";
}

}
