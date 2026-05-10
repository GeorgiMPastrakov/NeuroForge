#include "neuroforge/optim/SGD.hpp"

#include <stdexcept>
#include <utility>

namespace neuroforge {

SGD::SGD(std::vector<Parameter*> parameters, double learning_rate)
    : parameters_(std::move(parameters)), learning_rate_(learning_rate) {
    if (parameters_.empty()) {
        throw std::invalid_argument("SGD requires at least one parameter.");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("SGD learning rate must be positive.");
    }

    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            throw std::invalid_argument("SGD parameter list must not contain null parameters.");
        }
    }
}

void SGD::step() {
    for (Parameter* parameter : parameters_) {
        if (parameter->data().shape() != parameter->grad().shape() || parameter->data().size() != parameter->grad().size()) {
            throw std::invalid_argument("SGD gradient shape must match parameter data shape.");
        }

        for (size_t index = 0; index < parameter->data().size(); ++index) {
            parameter->data().data()[index] -= learning_rate_ * parameter->grad().data()[index];
        }
    }
}

void SGD::zeroGrad() {
    for (Parameter* parameter : parameters_) {
        parameter->zero_grad();
    }
}

}
