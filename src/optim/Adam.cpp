#include "neuroforge/optim/Adam.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace neuroforge {

Adam::Adam(std::vector<Parameter*> parameters, double learning_rate, double beta1, double beta2, double epsilon)
    : parameters_(std::move(parameters)),
      learning_rate_(learning_rate),
      beta1_(beta1),
      beta2_(beta2),
      epsilon_(epsilon) {
    if (parameters_.empty()) {
        throw std::invalid_argument("Adam requires at least one parameter.");
    }

    if (learning_rate <= 0.0) {
        throw std::invalid_argument("Adam learning rate must be positive.");
    }

    if (beta1 < 0.0 || beta1 >= 1.0) {
        throw std::invalid_argument("Adam beta1 must be in [0, 1).");
    }

    if (beta2 < 0.0 || beta2 >= 1.0) {
        throw std::invalid_argument("Adam beta2 must be in [0, 1).");
    }

    if (epsilon <= 0.0) {
        throw std::invalid_argument("Adam epsilon must be positive.");
    }

    first_moment_.reserve(parameters_.size());
    second_moment_.reserve(parameters_.size());

    for (Parameter* parameter : parameters_) {
        if (parameter == nullptr) {
            throw std::invalid_argument("Adam parameter list must not contain null parameters.");
        }

        first_moment_.push_back(std::vector<double>(parameter->data().size(), 0.0));
        second_moment_.push_back(std::vector<double>(parameter->data().size(), 0.0));
    }
}

void Adam::step() {
    ++step_count_;

    for (size_t parameter_index = 0; parameter_index < parameters_.size(); ++parameter_index) {
        Parameter* parameter = parameters_[parameter_index];

        if (parameter->data().shape() != parameter->grad().shape() || parameter->data().size() != parameter->grad().size()) {
            throw std::invalid_argument("Adam gradient shape must match parameter data shape.");
        }

        if (first_moment_[parameter_index].size() != parameter->data().size() || second_moment_[parameter_index].size() != parameter->data().size()) {
            throw std::invalid_argument("Adam optimizer state size must match parameter data size.");
        }

        const double first_bias = 1.0 - std::pow(beta1_, static_cast<double>(step_count_));
        const double second_bias = 1.0 - std::pow(beta2_, static_cast<double>(step_count_));

        for (size_t index = 0; index < parameter->data().size(); ++index) {
            const double gradient = parameter->grad().data()[index];
            first_moment_[parameter_index][index] = beta1_ * first_moment_[parameter_index][index] + (1.0 - beta1_) * gradient;
            second_moment_[parameter_index][index] = beta2_ * second_moment_[parameter_index][index] + (1.0 - beta2_) * gradient * gradient;
            const double corrected_first = first_moment_[parameter_index][index] / first_bias;
            const double corrected_second = second_moment_[parameter_index][index] / second_bias;
            parameter->data().data()[index] -= learning_rate_ * corrected_first / (std::sqrt(corrected_second) + epsilon_);
        }
    }
}

void Adam::zeroGrad() {
    for (Parameter* parameter : parameters_) {
        parameter->zero_grad();
    }
}

}
