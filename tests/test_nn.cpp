#include "neuroforge/nn/Module.hpp"
#include "neuroforge/nn/Parameter.hpp"

#include <cassert>
#include <string>
#include <vector>

using namespace neuroforge;

class EmptyModule : public Module {
public:
    Tensor forward(const Tensor& input) override {
        return input;
    }

    Tensor backward(const Tensor& grad_output) override {
        return grad_output;
    }

    std::string name() const override {
        return "EmptyModule";
    }
};

int main() {
    Parameter parameter(Tensor::fromVector({1.0, 2.0}), "weights");
    assert(parameter.name() == "weights");
    assert(parameter.data().shape() == Shape({2}));
    assert(parameter.grad().shape() == Shape({2}));
    assert(parameter.grad().at(0) == 0.0);
    assert(parameter.grad().at(1) == 0.0);

    parameter.grad().at(0) = 4.0;
    parameter.grad().at(1) = 5.0;
    parameter.zero_grad();
    assert(parameter.grad().at(0) == 0.0);
    assert(parameter.grad().at(1) == 0.0);

    EmptyModule module;
    assert(module.parameters().empty());
    module.zero_grad();
    assert(module.name() == "EmptyModule");

    return 0;
}
