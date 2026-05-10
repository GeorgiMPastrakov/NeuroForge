#pragma once

#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/losses/LossFunction.hpp"
#include "neuroforge/nn/Module.hpp"
#include "neuroforge/optim/Optimizer.hpp"
#include "neuroforge/training/TrainingConfig.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

namespace neuroforge {

class Trainer {
public:
    Trainer(Module& model, LossFunction& loss, Optimizer& optimizer);

    TrainingHistory fit(const Tensor& input, const Tensor& target, const TrainingConfig& config);
    double evaluateLoss(const Tensor& input, const Tensor& target);

private:
    Module& model_;
    LossFunction& loss_;
    Optimizer& optimizer_;
};

}
