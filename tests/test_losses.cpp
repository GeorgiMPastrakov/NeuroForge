#include "neuroforge/losses/BinaryCrossEntropyLoss.hpp"
#include "neuroforge/losses/CrossEntropyLoss.hpp"
#include "neuroforge/losses/MAELoss.hpp"
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

    MAELoss mae;
    assert(mae.name() == "MAELoss");
    Tensor mae_value = mae.forward(prediction, target);
    assert(near(mae_value.item(), 1.0));
    Tensor mae_grad = mae.backward();
    assert(near(mae_grad.at(0), 0.0));
    assert(near(mae_grad.at(1), 1.0 / 3.0));
    assert(near(mae_grad.at(2), 1.0 / 3.0));
    expectThrows<std::invalid_argument>([] { MAELoss().backward(); });
    expectThrows<std::invalid_argument>([] {
        MAELoss invalid;
        invalid.forward(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });

    Tensor mae_autograd_prediction(std::vector<double>{1.0, 3.0}, Shape({2}), true);
    Tensor mae_autograd_value = mae.forward(mae_autograd_prediction, Tensor::fromVector({2.0, 1.0}));
    mae_autograd_value.backward();
    assert(near(mae_autograd_prediction.grad()[0], -0.5));
    assert(near(mae_autograd_prediction.grad()[1], 0.5));

    BinaryCrossEntropyLoss bce;
    assert(bce.name() == "BinaryCrossEntropyLoss");
    Tensor bce_prediction = Tensor::fromVector({0.25, 0.75});
    Tensor bce_target = Tensor::fromVector({0.0, 1.0});
    Tensor bce_value = bce.forward(bce_prediction, bce_target);
    assert(near(bce_value.item(), -std::log(0.75)));
    Tensor bce_grad = bce.backward();
    assert(near(bce_grad.at(0), 2.0 / 3.0));
    assert(near(bce_grad.at(1), -2.0 / 3.0));
    assert(std::isfinite(bce.forward(Tensor::fromVector({0.0, 1.0}), bce_target).item()));
    expectThrows<std::invalid_argument>([] { BinaryCrossEntropyLoss().backward(); });
    expectThrows<std::invalid_argument>([] {
        BinaryCrossEntropyLoss invalid;
        invalid.forward(Tensor::fromVector({1.0}), Tensor::fromVector({1.0, 2.0}));
    });

    Tensor bce_autograd_prediction(std::vector<double>{0.25, 0.75}, Shape({2}), true);
    Tensor bce_autograd_value = bce.forward(bce_autograd_prediction, bce_target);
    bce_autograd_value.backward();
    assert(near(bce_autograd_prediction.grad()[0], 2.0 / 3.0));
    assert(near(bce_autograd_prediction.grad()[1], -2.0 / 3.0));

    CrossEntropyLoss cross_entropy;
    assert(cross_entropy.name() == "CrossEntropyLoss");
    Tensor logits = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 1.0}
    });
    Tensor classes = Tensor::fromVector({
        {0.0, 1.0},
        {1.0, 0.0}
    });
    Tensor ce_value = cross_entropy.forward(logits, classes);
    Tensor probabilities = logits.softmaxRows();
    const double expected_ce = (-std::log(probabilities.at(0, 1)) - std::log(probabilities.at(1, 0))) / 2.0;
    assert(near(ce_value.item(), expected_ce));
    Tensor ce_grad = cross_entropy.backward();
    assert(near(ce_grad.at(0, 0), probabilities.at(0, 0) / 2.0));
    assert(near(ce_grad.at(0, 1), (probabilities.at(0, 1) - 1.0) / 2.0));
    assert(near(ce_grad.at(1, 0), (probabilities.at(1, 0) - 1.0) / 2.0));
    assert(near(ce_grad.at(1, 1), probabilities.at(1, 1) / 2.0));
    expectThrows<std::invalid_argument>([] { CrossEntropyLoss().backward(); });
    expectThrows<std::invalid_argument>([] {
        CrossEntropyLoss invalid;
        invalid.forward(Tensor::fromVector({1.0}), Tensor::fromVector({1.0}));
    });
    expectThrows<std::invalid_argument>([] {
        CrossEntropyLoss invalid;
        invalid.forward(Tensor::fromVector({{1.0, 2.0}}), Tensor::fromVector({{1.0}}));
    });

    Tensor ce_autograd_logits(std::vector<double>{1.0, 2.0}, Shape({1, 2}), true);
    Tensor ce_autograd_value = cross_entropy.forward(ce_autograd_logits, Tensor::fromVector({{0.0, 1.0}}));
    ce_autograd_value.backward();
    Tensor ce_autograd_probabilities = Tensor::fromVector({{1.0, 2.0}}).softmaxRows();
    assert(near(ce_autograd_logits.grad()[0], ce_autograd_probabilities.at(0, 0)));
    assert(near(ce_autograd_logits.grad()[1], ce_autograd_probabilities.at(0, 1) - 1.0));

    return 0;
}
