#include "neuroforge/nn/Parameter.hpp"
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

    return 0;
}
