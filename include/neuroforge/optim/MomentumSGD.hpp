#pragma once

#include "neuroforge/nn/Parameter.hpp"
#include "neuroforge/optim/Optimizer.hpp"

#include <vector>

namespace neuroforge {

class MomentumSGD : public Optimizer {
public:
    MomentumSGD(std::vector<Parameter*> parameters, double learning_rate, double momentum = 0.9);

    void step() override;
    void zeroGrad() override;

private:
    std::vector<Parameter*> parameters_;
    std::vector<std::vector<double>> velocity_;
    double learning_rate_;
    double momentum_;
};

}
