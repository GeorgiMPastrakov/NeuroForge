#include "neuroforge/nn/Parameter.hpp"
#include "neuroforge/optim/Adam.hpp"
#include "neuroforge/optim/MomentumSGD.hpp"
#include "neuroforge/optim/SGD.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace neuroforge;

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
    parameter.grad().at(0) = 0.5;
    parameter.grad().at(1) = -1.0;

    SGD optimizer({&parameter}, 0.1);
    optimizer.step();
    assert(near(parameter.data().at(0), 0.95));
    assert(near(parameter.data().at(1), 2.1));

    optimizer.zeroGrad();
    assert(parameter.grad().at(0) == 0.0);
    assert(parameter.grad().at(1) == 0.0);

    expectThrows<std::invalid_argument>([] { SGD({}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        SGD optimizer({&parameter}, 0.0);
    });
    expectThrows<std::invalid_argument>([] { SGD({nullptr}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        parameter.grad().data().push_back(1.0);
        SGD optimizer({&parameter}, 0.1);
        optimizer.step();
    });

    Parameter momentum_parameter(Tensor::fromVector({1.0, 2.0}), "weights");
    momentum_parameter.grad().at(0) = 0.5;
    momentum_parameter.grad().at(1) = -1.0;
    MomentumSGD momentum_optimizer({&momentum_parameter}, 0.1, 0.5);
    momentum_optimizer.step();
    assert(near(momentum_parameter.data().at(0), 0.95));
    assert(near(momentum_parameter.data().at(1), 2.1));
    momentum_parameter.grad().at(0) = 0.5;
    momentum_parameter.grad().at(1) = -1.0;
    momentum_optimizer.step();
    assert(near(momentum_parameter.data().at(0), 0.875));
    assert(near(momentum_parameter.data().at(1), 2.25));
    momentum_optimizer.zeroGrad();
    assert(momentum_parameter.grad().at(0) == 0.0);
    assert(momentum_parameter.grad().at(1) == 0.0);

    expectThrows<std::invalid_argument>([] { MomentumSGD({}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        MomentumSGD optimizer({&parameter}, 0.0);
    });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        MomentumSGD optimizer({&parameter}, 0.1, 1.0);
    });
    expectThrows<std::invalid_argument>([] { MomentumSGD({nullptr}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        parameter.grad().data().push_back(1.0);
        MomentumSGD optimizer({&parameter}, 0.1);
        optimizer.step();
    });

    Parameter adam_parameter(Tensor::fromVector({1.0, 2.0}), "weights");
    adam_parameter.grad().at(0) = 0.5;
    adam_parameter.grad().at(1) = -1.0;
    Adam adam_optimizer({&adam_parameter}, 0.1);
    adam_optimizer.step();
    assert(near(adam_parameter.data().at(0), 0.900000002));
    assert(near(adam_parameter.data().at(1), 2.099999999));
    adam_optimizer.zeroGrad();
    assert(adam_parameter.grad().at(0) == 0.0);
    assert(adam_parameter.grad().at(1) == 0.0);

    expectThrows<std::invalid_argument>([] { Adam({}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        Adam optimizer({&parameter}, 0.0);
    });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        Adam optimizer({&parameter}, 0.1, 1.0);
    });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        Adam optimizer({&parameter}, 0.1, 0.9, 1.0);
    });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        Adam optimizer({&parameter}, 0.1, 0.9, 0.999, 0.0);
    });
    expectThrows<std::invalid_argument>([] { Adam({nullptr}, 0.1); });
    expectThrows<std::invalid_argument>([] {
        Parameter parameter(Tensor::fromVector({1.0}), "weight");
        parameter.grad().data().push_back(1.0);
        Adam optimizer({&parameter}, 0.1);
        optimizer.step();
    });

    return 0;
}
