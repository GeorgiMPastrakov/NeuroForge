#include "neuroforge/nn/Parameter.hpp"

#include <utility>

namespace neuroforge {

Parameter::Parameter(Tensor data, std::string name)
    : data_(std::move(data)), grad_(Tensor::zeros(data_.shape())), name_(std::move(name)) {
}

Tensor& Parameter::data() {
    return data_;
}

const Tensor& Parameter::data() const {
    return data_;
}

Tensor& Parameter::grad() {
    return grad_;
}

const Tensor& Parameter::grad() const {
    return grad_;
}

const std::string& Parameter::name() const {
    return name_;
}

void Parameter::zero_grad() {
    grad_ = Tensor::zeros(data_.shape());
}

}
