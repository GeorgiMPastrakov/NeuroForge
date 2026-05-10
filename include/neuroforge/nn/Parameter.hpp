#pragma once

#include "neuroforge/core/Tensor.hpp"

#include <string>

namespace neuroforge {

class Parameter {
public:
    explicit Parameter(Tensor data, std::string name = "");

    Tensor& data();
    const Tensor& data() const;
    Tensor& grad();
    const Tensor& grad() const;
    const std::string& name() const;
    void zero_grad();
    void enableAutograd();
    void syncAutogradGrad();

private:
    Tensor data_;
    Tensor grad_;
    std::string name_;
};

}
