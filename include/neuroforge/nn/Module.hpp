#pragma once

#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/nn/Parameter.hpp"

#include <string>
#include <vector>

namespace neuroforge {

class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual Tensor backward(const Tensor& grad_output) = 0;
    virtual std::vector<Parameter*> parameters();
    virtual void zero_grad();
    virtual void enableAutograd();
    virtual void syncAutogradGradients();
    virtual std::string name() const = 0;
    virtual ~Module() = default;
};

}
