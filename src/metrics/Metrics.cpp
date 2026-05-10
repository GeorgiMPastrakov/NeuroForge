#include "neuroforge/metrics/Metrics.hpp"

#include <cmath>
#include <stdexcept>

namespace neuroforge {

namespace {

void requireSameShape(const Tensor& prediction, const Tensor& target, const std::string& name) {
    if (prediction.shape() != target.shape()) {
        throw std::invalid_argument(name + " shape mismatch: prediction shape " + prediction.shape().toString() + " does not match target shape " + target.shape().toString() + ".");
    }
}

}

double mse(const Tensor& prediction, const Tensor& target) {
    requireSameShape(prediction, target, "MSE");

    double total = 0.0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        const double diff = prediction.data()[index] - target.data()[index];
        total += diff * diff;
    }

    return total / static_cast<double>(prediction.size());
}

double mae(const Tensor& prediction, const Tensor& target) {
    requireSameShape(prediction, target, "MAE");

    double total = 0.0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        total += std::fabs(prediction.data()[index] - target.data()[index]);
    }

    return total / static_cast<double>(prediction.size());
}

double accuracy(const Tensor& prediction, const Tensor& target) {
    requireSameShape(prediction, target, "Accuracy");

    size_t correct = 0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        if (prediction.data()[index] == target.data()[index]) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(prediction.size());
}

double binaryAccuracy(const Tensor& prediction, const Tensor& target) {
    requireSameShape(prediction, target, "BinaryAccuracy");

    size_t correct = 0;

    for (size_t index = 0; index < prediction.size(); ++index) {
        const bool predicted = prediction.data()[index] >= 0.5;
        const bool expected = target.data()[index] >= 0.5;

        if (predicted == expected) {
            ++correct;
        }
    }

    return static_cast<double>(correct) / static_cast<double>(prediction.size());
}

}
