#pragma once

#include "neuroforge/nn/Module.hpp"

#include <optional>
#include <string>

namespace neuroforge {

class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    Tensor backward(const Tensor& grad_output) override;
    std::string name() const override;

private:
    std::optional<Tensor> cached_input_;
};

}
