#pragma once

#include "neuroforge/core/Tensor.hpp"

#include <string>

namespace neuroforge {

class LossFunction {
public:
    virtual Tensor forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward() = 0;
    virtual std::string name() const = 0;
    virtual ~LossFunction() = default;
};

}
