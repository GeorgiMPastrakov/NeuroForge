#pragma once

#include "neuroforge/losses/LossFunction.hpp"

#include <optional>
#include <string>

namespace neuroforge {

class CrossEntropyLoss : public LossFunction {
public:
    Tensor forward(const Tensor& prediction, const Tensor& target) override;
    Tensor backward() override;
    std::string name() const override;

private:
    std::optional<Tensor> probabilities_;
    std::optional<Tensor> target_;
};

}
