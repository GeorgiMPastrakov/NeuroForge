#include "neuroforge/training/TrainingConfig.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

#include <cassert>
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

    return 0;
}
