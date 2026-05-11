#include "neuroforge/nn/Module.hpp"
#include "neuroforge/nn/Parameter.hpp"
#include "neuroforge/nn/Dropout.hpp"
#include "neuroforge/nn/LeakyReLU.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/nn/Softmax.hpp"
#include "neuroforge/nn/Tanh.hpp"

#include <cassert>
#include <cmath>
#include <memory>
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

class OrderModule : public Module {
public:
    OrderModule(int value, std::vector<int>& order)
        : value_(value), order_(order) {
    }

    Tensor forward(const Tensor& input) override {
        return input;
    }

    Tensor backward(const Tensor& grad_output) override {
        order_.push_back(value_);
        return grad_output;
    }

    std::string name() const override {
        return "OrderModule";
    }

private:
    int value_;
    std::vector<int>& order_;
};

template <typename Exception, typename Function>
void expectThrows(Function function) {
    bool thrown = false;

    try {
        function();
    } catch (const Exception&) {
        thrown = true;
    }

    assert(thrown);
}

bool near(double left, double right) {
    return std::fabs(left - right) < 1e-9;
}

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
    assert(module.isTraining());
    module.eval();
    assert(!module.isTraining());
    module.train();
    assert(module.isTraining());

    Linear linear(2, 3);
    assert(linear.inFeatures() == 2);
    assert(linear.outFeatures() == 3);
    assert(linear.parameterCount() == 9);
    assert(linear.parameters().size() == 2);
    assert(linear.weights().data().shape() == Shape({2, 3}));
    assert(linear.bias().data().shape() == Shape({1, 3}));
    assert(linear.name() == "Linear(2 -> 3)");

    linear.weights().data().at(0, 0) = 1.0;
    linear.weights().data().at(0, 1) = 2.0;
    linear.weights().data().at(0, 2) = 3.0;
    linear.weights().data().at(1, 0) = 4.0;
    linear.weights().data().at(1, 1) = 5.0;
    linear.weights().data().at(1, 2) = 6.0;
    linear.bias().data().at(0, 0) = 0.5;
    linear.bias().data().at(0, 1) = 1.0;
    linear.bias().data().at(0, 2) = -1.0;

    Tensor input = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0}
    });
    Tensor output = linear.forward(input);
    assert(output.shape() == Shape({2, 3}));
    assert(near(output.at(0, 0), 9.5));
    assert(near(output.at(0, 1), 13.0));
    assert(near(output.at(0, 2), 14.0));
    assert(near(output.at(1, 0), 19.5));
    assert(near(output.at(1, 1), 27.0));
    assert(near(output.at(1, 2), 32.0));

    Tensor grad_output = Tensor::fromVector({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    });
    Tensor input_grad = linear.backward(grad_output);
    assert(input_grad.shape() == Shape({2, 2}));
    assert(near(input_grad.at(0, 0), 14.0));
    assert(near(input_grad.at(0, 1), 32.0));
    assert(near(input_grad.at(1, 0), 32.0));
    assert(near(input_grad.at(1, 1), 77.0));
    assert(near(linear.weights().grad().at(0, 0), 13.0));
    assert(near(linear.weights().grad().at(0, 1), 17.0));
    assert(near(linear.weights().grad().at(0, 2), 21.0));
    assert(near(linear.weights().grad().at(1, 0), 18.0));
    assert(near(linear.weights().grad().at(1, 1), 24.0));
    assert(near(linear.weights().grad().at(1, 2), 30.0));
    assert(near(linear.bias().grad().at(0, 0), 5.0));
    assert(near(linear.bias().grad().at(0, 1), 7.0));
    assert(near(linear.bias().grad().at(0, 2), 9.0));

    linear.zero_grad();
    assert(near(linear.weights().grad().at(0, 0), 0.0));
    assert(near(linear.bias().grad().at(0, 0), 0.0));

    expectThrows<std::invalid_argument>([] { Linear(0, 1); });
    expectThrows<std::invalid_argument>([] { Linear(1, 0); });
    expectThrows<std::invalid_argument>([] { Linear(2, 1).forward(Tensor::fromVector({1.0, 2.0})); });
    expectThrows<std::invalid_argument>([] { Linear(2, 1).forward(Tensor::fromVector({{1.0, 2.0, 3.0}})); });
    expectThrows<std::invalid_argument>([] { Linear(2, 1).backward(Tensor::fromVector({{1.0}})); });

    Tensor activation_input = Tensor::fromVector({-1.0, 0.0, 1.0});
    Tensor activation_grad = Tensor::fromVector({2.0, 3.0, 4.0});

    ReLU relu_module;
    Tensor relu_output = relu_module.forward(activation_input);
    assert(relu_module.name() == "ReLU()");
    assert(relu_module.parameters().empty());
    assert(relu_output.at(0) == 0.0);
    assert(relu_output.at(1) == 0.0);
    assert(relu_output.at(2) == 1.0);
    Tensor relu_grad = relu_module.backward(activation_grad);
    assert(relu_grad.at(0) == 0.0);
    assert(relu_grad.at(1) == 0.0);
    assert(relu_grad.at(2) == 4.0);

    Sigmoid sigmoid_module;
    Tensor sigmoid_output = sigmoid_module.forward(activation_input);
    assert(sigmoid_module.name() == "Sigmoid()");
    assert(sigmoid_module.parameters().empty());
    assert(near(sigmoid_output.at(1), 0.5));
    Tensor sigmoid_grad = sigmoid_module.backward(activation_grad);
    assert(near(sigmoid_grad.at(1), 0.75));

    Tanh tanh_module;
    Tensor tanh_output = tanh_module.forward(activation_input);
    assert(tanh_module.name() == "Tanh()");
    assert(tanh_module.parameters().empty());
    assert(near(tanh_output.at(1), 0.0));
    Tensor tanh_grad = tanh_module.backward(activation_grad);
    assert(near(tanh_grad.at(1), 3.0));

    LeakyReLU leaky_relu_module(0.1);
    Tensor leaky_relu_output = leaky_relu_module.forward(activation_input);
    assert(leaky_relu_module.name() == "LeakyReLU(0.1)");
    assert(leaky_relu_module.parameters().empty());
    assert(near(leaky_relu_output.at(0), -0.1));
    assert(near(leaky_relu_output.at(2), 1.0));
    Tensor leaky_relu_grad = leaky_relu_module.backward(activation_grad);
    assert(near(leaky_relu_grad.at(0), 0.2));
    assert(near(leaky_relu_grad.at(1), 0.3));
    assert(near(leaky_relu_grad.at(2), 4.0));

    Softmax softmax_module;
    Tensor softmax_input = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 3.0}
    });
    Tensor softmax_output = softmax_module.forward(softmax_input);
    assert(softmax_module.name() == "Softmax()");
    assert(near(softmax_output.at(0, 0) + softmax_output.at(0, 1), 1.0));
    assert(near(softmax_output.at(1, 0), 0.5));
    Tensor softmax_grad = softmax_module.backward(Tensor::fromVector({
        {3.0, 5.0},
        {1.0, 1.0}
    }));
    const double s0 = softmax_output.at(0, 0);
    const double s1 = softmax_output.at(0, 1);
    const double weighted = 3.0 * s0 + 5.0 * s1;
    assert(near(softmax_grad.at(0, 0), s0 * (3.0 - weighted)));
    assert(near(softmax_grad.at(0, 1), s1 * (5.0 - weighted)));
    assert(near(softmax_grad.at(1, 0), 0.0));
    assert(near(softmax_grad.at(1, 1), 0.0));

    Dropout dropout_module(0.5, 42);
    Tensor dropout_input = Tensor::ones(Shape({4}));
    Tensor dropout_output = dropout_module.forward(dropout_input);
    size_t dropped = 0;
    size_t kept = 0;

    for (double value : dropout_output.data()) {
        if (near(value, 0.0)) {
            ++dropped;
        } else if (near(value, 2.0)) {
            ++kept;
        }
    }

    assert(dropped + kept == dropout_output.size());
    Tensor dropout_grad = dropout_module.backward(Tensor::ones(Shape({4})));
    assert(dropout_grad.data() == dropout_output.data());
    dropout_module.eval();
    Tensor dropout_eval_output = dropout_module.forward(dropout_input);
    assert(dropout_eval_output.data() == dropout_input.data());
    Tensor dropout_eval_grad = dropout_module.backward(Tensor::ones(Shape({4})));
    assert(dropout_eval_grad.data() == Tensor::ones(Shape({4})).data());

    expectThrows<std::invalid_argument>([] { ReLU().backward(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { Sigmoid().backward(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { Tanh().backward(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { LeakyReLU().backward(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { Softmax().backward(Tensor::fromVector({1.0})); });
    expectThrows<std::invalid_argument>([] { Dropout(1.0); });
    expectThrows<std::invalid_argument>([] {
        Dropout dropout(0.5);
        dropout.backward(Tensor::fromVector({1.0}));
    });
    expectThrows<std::invalid_argument>([] {
        ReLU relu;
        relu.forward(Tensor::fromVector({1.0}));
        relu.backward(Tensor::fromVector({1.0, 2.0}));
    });

    Sequential sequential;
    sequential.add(std::make_shared<Linear>(2, 4));
    sequential.add(std::make_shared<ReLU>());
    sequential.add(std::make_shared<Linear>(4, 1));
    sequential.add(std::make_shared<Sigmoid>());
    assert(sequential.name() == "Sequential");
    assert(sequential.size() == 4);
    assert(sequential.parameters().size() == 4);
    assert(sequential.isTraining());
    sequential.eval();
    assert(!sequential.isTraining());
    for (const auto& layer : sequential.layers()) {
        assert(!layer->isTraining());
    }
    sequential.train();
    assert(sequential.isTraining());
    for (const auto& layer : sequential.layers()) {
        assert(layer->isTraining());
    }
    Tensor sequential_output = sequential.forward(Tensor::fromVector({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    }));
    assert(sequential_output.shape() == Shape({4, 1}));
    std::string summary = sequential.summary();
    assert(summary.find("Linear(2 -> 4)") != std::string::npos);
    assert(summary.find("ReLU()") != std::string::npos);
    assert(summary.find("Sigmoid()") != std::string::npos);
    assert(summary.find("Total parameters: 17") != std::string::npos);

    std::vector<int> order;
    Sequential ordered;
    ordered.add(std::make_shared<OrderModule>(1, order));
    ordered.add(std::make_shared<OrderModule>(2, order));
    ordered.add(std::make_shared<OrderModule>(3, order));
    ordered.forward(Tensor::fromVector({1.0}));
    ordered.backward(Tensor::fromVector({1.0}));
    assert(order.size() == 3);
    assert(order[0] == 3);
    assert(order[1] == 2);
    assert(order[2] == 1);

    expectThrows<std::invalid_argument>([] {
        Sequential invalid;
        invalid.add(nullptr);
    });

    return 0;
}
