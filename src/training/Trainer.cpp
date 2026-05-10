#include "neuroforge/training/Trainer.hpp"

#include <stdexcept>

namespace neuroforge {

Trainer::Trainer(Module& model, LossFunction& loss, Optimizer& optimizer)
    : model_(model), loss_(loss), optimizer_(optimizer) {
}

TrainingHistory Trainer::fit(const Tensor& input, const Tensor& target, const TrainingConfig& config) {
    if (config.epochs == 0) {
        throw std::invalid_argument("TrainingConfig epochs must be greater than zero.");
    }

    TrainingHistory history;

    for (size_t epoch = 1; epoch <= config.epochs; ++epoch) {
        optimizer_.zeroGrad();
        Tensor prediction = model_.forward(input);
        Tensor loss_value = loss_.forward(prediction, target);
        Tensor grad = loss_.backward();
        model_.backward(grad);
        optimizer_.step();
        history.addLoss(epoch, loss_value.item());
    }

    return history;
}

double Trainer::evaluateLoss(const Tensor& input, const Tensor& target) {
    Tensor prediction = model_.forward(input);
    return loss_.forward(prediction, target).item();
}

}
