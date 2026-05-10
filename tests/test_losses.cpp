#include "neuroforge/losses/MSELoss.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

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
    MSELoss loss;
    assert(loss.name() == "MSELoss");

    Tensor prediction = Tensor::fromVector({1.0, 2.0, 4.0});
    Tensor target = Tensor::fromVector({1.0, 1.0, 2.0});
    Tensor value = loss.forward(prediction, target);
    assert(value.shape() == Shape({1}));
    assert(near(value.item(), 5.0 / 3.0));

    Tensor grad = loss.backward();
    assert(grad.shape() == Shape({3}));
    assert(near(grad.at(0), 0.0));
    assert(near(grad.at(1), 2.0 / 3.0));
    assert(near(grad.at(2), 4.0 / 3.0));

    expectThrows<std::invalid_argument>([] { MSELoss().backward(); });
    expectThrows<std::invalid_argument>([] {
        MSELoss invalid;
        invalid.forward(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });

    return 0;
}
