#pragma once

#include "neuroforge/nn/Parameter.hpp"
#include "neuroforge/optim/Optimizer.hpp"

#include <cstddef>
#include <vector>

namespace neuroforge {

class Adam : public Optimizer {
public:
    Adam(std::vector<Parameter*> parameters, double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);

    void step() override;
    void zeroGrad() override;

private:
    std::vector<Parameter*> parameters_;
    std::vector<std::vector<double>> first_moment_;
    std::vector<std::vector<double>> second_moment_;
    double learning_rate_;
    double beta1_;
    double beta2_;
    double epsilon_;
    size_t step_count_ = 0;
};

}
