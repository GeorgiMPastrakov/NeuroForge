#include "neuroforge/training/TrainingConfig.hpp"
#include "neuroforge/training/TrainingHistory.hpp"
#include "neuroforge/training/Trainer.hpp"
#include "neuroforge/losses/MSELoss.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/optim/SGD.hpp"

#include <cassert>
#include <cmath>
#include <memory>
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

int main() {
    TrainingConfig config;
    assert(config.epochs == 1000);
    assert(config.log_interval == 100);
    assert(config.verbose == false);

    TrainingHistory history;
    assert(history.empty());
    assert(history.size() == 0);
    expectThrows<std::invalid_argument>([&] { history.finalLoss(); });

    history.addLoss(1, 0.5);
    history.addLoss(2, 0.25);
    assert(!history.empty());
    assert(history.size() == 2);
    assert(history.epochs()[0] == 1);
    assert(history.epochs()[1] == 2);
    assert(history.losses()[0] == 0.5);
    assert(history.losses()[1] == 0.25);
    assert(history.finalLoss() == 0.25);

    Linear linear(1, 1);
    linear.weights().data().at(0, 0) = 0.0;
    linear.bias().data().at(0, 0) = 0.0;

    MSELoss loss;
    SGD optimizer(linear.parameters(), 0.1);
    Trainer trainer(linear, loss, optimizer);

    TrainingConfig one_epoch;
    one_epoch.epochs = 1;
    Tensor input = Tensor::fromVector({
        {1.0},
        {2.0}
    });
    Tensor target = Tensor::fromVector({
        {2.0},
        {4.0}
    });
    double before = trainer.evaluateLoss(input, target);
    TrainingHistory one_epoch_history = trainer.fit(input, target, one_epoch);
    double after = trainer.evaluateLoss(input, target);
    assert(one_epoch_history.size() == 1);
    assert(one_epoch_history.epochs()[0] == 1);
    assert(linear.weights().data().at(0, 0) != 0.0);
    assert(linear.bias().data().at(0, 0) != 0.0);
    assert(after < before);

    double weight_after_train = linear.weights().data().at(0, 0);
    double bias_after_train = linear.bias().data().at(0, 0);
    double evaluated = trainer.evaluateLoss(input, target);
    assert(evaluated >= 0.0);
    assert(linear.weights().data().at(0, 0) == weight_after_train);
    assert(linear.bias().data().at(0, 0) == bias_after_train);

    expectThrows<std::invalid_argument>([&] {
        TrainingConfig invalid;
        invalid.epochs = 0;
        trainer.fit(input, target, invalid);
    });

    return 0;
}
