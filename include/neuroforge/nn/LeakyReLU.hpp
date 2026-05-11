#pragma once

#include "neuroforge/nn/Module.hpp"

#include <optional>
#include <string>

namespace neuroforge {

class LeakyReLU : public Module {
public:
    explicit LeakyReLU(double negative_slope = 0.01);

    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override;
    double negativeSlope() const;

private:
    double negative_slope_;
    std::optional<Tensor> cached_input_;
};

}
