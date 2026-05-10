#pragma once

#include "neuroforge/nn/Parameter.hpp"
#include "neuroforge/optim/Optimizer.hpp"

#include <vector>

namespace neuroforge {

class SGD : public Optimizer {
public:
    SGD(std::vector<Parameter*> parameters, double learning_rate);

    void step() override;
    void zeroGrad() override;

private:
    std::vector<Parameter*> parameters_;
    double learning_rate_;
};

}
