#include "neuroforge/metrics/Metrics.hpp"

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
    Tensor prediction = Tensor::fromVector({
        {1.0},
        {3.0},
        {5.0}
    });
    Tensor target = Tensor::fromVector({
        {1.0},
        {2.0},
        {1.0}
    });

    assert(near(mse(prediction, target), 17.0 / 3.0));
    assert(near(mae(prediction, target), 5.0 / 3.0));

    Tensor classes = Tensor::fromVector({1.0, 0.0, 1.0, 1.0});
    Tensor class_target = Tensor::fromVector({1.0, 1.0, 1.0, 0.0});
    assert(near(accuracy(classes, class_target), 0.5));

    Tensor probabilities = Tensor::fromVector({0.1, 0.5, 0.9, 0.4});
    Tensor binary_target = Tensor::fromVector({0.0, 1.0, 1.0, 1.0});
    assert(near(binaryAccuracy(probabilities, binary_target), 0.75));

    expectThrows<std::invalid_argument>([] {
        mse(Tensor::fromVector({1.0}), Tensor::fromVector({{1.0}}));
    });
    expectThrows<std::invalid_argument>([] {
        mae(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });
    expectThrows<std::invalid_argument>([] {
        accuracy(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });
    expectThrows<std::invalid_argument>([] {
        binaryAccuracy(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });

    return 0;
}
