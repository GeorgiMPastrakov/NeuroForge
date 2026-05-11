#include "neuroforge/optim/MomentumSGD.hpp"

#include <stdexcept>
#include <utility>

namespace neuroforge {

MomentumSGD::MomentumSGD(std::vector<Parameter*> parameters, double learning_rate, double momentum)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate), momentum_(momentum) {
    if (parameters_.empty()) {
        throw std::invalid_argument("MomentumSGD requires at least one parameter.");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("MomentumSGD learning rate must be positive.");
    }

    if (momentum < 0.0 || momentum >= 1.0) {
        throw std::invalid_argument("MomentumSGD momentum must be in [0, 1).");
    }

    velocity_.reserve(parameters_.size());

    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            throw std::invalid_argument("MomentumSGD parameter list must not contain null parameters.");
        }

        velocity_.push_back(std::vector<double>(parameter->data().size(), 0.0));
    }
}

void MomentumSGD::step() {
    for (size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        Parameter* parameter = parameters_[parameter_index];

        if (parameter->data().shape() != parameter->grad().shape() || parameter->data().size() != parameter->grad().size()) {
            throw std::invalid_argument("MomentumSGD gradient shape must match parameter data shape.");
        }

        if (velocity_[parameter_index].size() != parameter->data().size()) {
            throw std::invalid_argument("MomentumSGD optimizer state size must match parameter data size.");
        }

        for (size_t index = 0; index < parameter->data().size(); ++index) {
            velocity_[parameter_index][index] = momentum_ * velocity_[parameter_index][index] - learning_rate_ * parameter->grad().data()[index];
            parameter->data().data()[index] += velocity_[parameter_index][index];
        }
    }
}

void MomentumSGD::zeroGrad() {
    for (Parameter* parameter : parameters_) {
        parameter->zero_grad();
    }
}

}
