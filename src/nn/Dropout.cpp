#include "neuroforge/nn/Dropout.hpp"

#include <random>
#include <sstream>
#include <stdexcept>

namespace neuroforge {

Dropout::Dropout(double probability, uint32_t seed)
    : probability_(probability), seed_(seed) {
    if (probability < 0.0 || probability >= 1.0) {
        throw std::invalid_argument("Dropout probability must be in [0, 1).");
    }
}

Tensor Dropout::forward(const Tensor& input) {
    if (!isTraining() || probability_ == 0.0) {
        cached_mask_ = Tensor::ones(input.shape());
        return input;
    }

    std::mt19937 generator(seed_ + static_cast<uint32_t>(forward_count_));
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    ++forward_count_;

    Tensor mask = Tensor::zeros(input.shape());
    const double scale = 1.0 / (1.0 - probability_);

    for (size_t index = 0; index < mask.size(); ++index) {
        mask.data()[index] = distribution(generator) >= probability_ ? scale : 0.0;
    }

    cached_mask_ = mask;
    return input.multiply(mask);
}

Tensor Dropout::backward(const Tensor& grad_output) {
    if (!cached_mask_.has_value()) {
        throw std::invalid_argument("Dropout backward requires a prior forward call.");
    }

    if (grad_output.shape() != cached_mask_->shape()) {
        throw std::invalid_argument("Dropout grad output shape mismatch: expected " + cached_mask_->shape().toString() + ", got " + grad_output.shape().toString() + ".");
    }

    return grad_output.multiply(cached_mask_.value());
}

std::string Dropout::name() const {
    std::ostringstream stream;
    stream << "Dropout(" << probability_ << ")";
    return stream.str();
}

}
