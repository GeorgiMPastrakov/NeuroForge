#include "neuroforge/visualization/VisualizationData.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

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

bool near(double left, double right) {
    return std::fabs(left - right) < 1e-9;
}

int main() {
    Sequential empty;
    ModelSnapshot empty_snapshot = buildModelSnapshot(empty);
    assert(empty_snapshot.layers.empty());
    assert(empty_snapshot.total_parameters == 0);

    Sequential model;
    auto linear = std::make_shared<Linear>(2, 1);
    linear->weights().data().at(0, 0) = 1.0;
    linear->weights().data().at(1, 0) = 2.0;
    linear->bias().data().at(0, 0) = 0.5;
    linear->weights().grad().at(0, 0) = 3.0;
    linear->weights().grad().at(1, 0) = 4.0;
    model.add(linear);
    model.add(std::make_shared<ReLU>());

    ModelSnapshot model_snapshot = buildModelSnapshot(model);
    assert(model_snapshot.layers.size() == 2);
    assert(model_snapshot.layers[0].name == "Linear(2 -> 1)");
    assert(model_snapshot.layers[0].parameter_count == 3);
    assert(model_snapshot.layers[1].name == "ReLU()");
    assert(model_snapshot.total_parameters == 3);

    TrainingHistory history;
    LossHistorySnapshot empty_history = buildLossHistorySnapshot(history);
    assert(empty_history.epochs.empty());
    assert(empty_history.losses.empty());
    history.addLoss(1, 0.5);
    history.addLoss(2, 0.25);
    LossHistorySnapshot history_snapshot = buildLossHistorySnapshot(history);
    assert(history_snapshot.epochs.size() == 2);
    assert(history_snapshot.losses[1] == 0.25);

    Tensor tensor = Tensor::fromVector({1.0, 2.0, 3.0});
    TensorSnapshot tensor_snapshot = buildTensorSnapshot("tensor", tensor, 2);
    assert(tensor_snapshot.name == "tensor");
    assert(tensor_snapshot.shape == Shape({3}));
    assert(tensor_snapshot.values.size() == 2);
    assert(tensor_snapshot.original_size == 3);
    assert(tensor_snapshot.truncated);

    std::vector<GradientSnapshot> gradients = buildGradientSnapshots(model, 10);
    assert(gradients.size() == 2);
    assert(gradients[0].name == "weights");
    assert(near(gradients[0].norm, 5.0));
    assert(gradients[1].name == "bias");
    assert(near(gradients[1].norm, 0.0));

    DecisionBoundaryGrid grid = buildDecisionBoundaryGrid(model, 0.0, 1.0, 0.0, 1.0, 2, 3);
    assert(grid.rows == 2);
    assert(grid.cols == 3);
    assert(grid.x_values.size() == 3);
    assert(grid.y_values.size() == 2);
    assert(grid.predictions.size() == 6);
    assert(near(grid.predictions.front(), 0.5));

    expectThrows<std::invalid_argument>([] {
        Tensor tensor = Tensor::fromVector({1.0});
        buildTensorSnapshot("bad", tensor, 0);
    });
    expectThrows<std::invalid_argument>([&] {
        buildDecisionBoundaryGrid(model, 0.0, 1.0, 0.0, 1.0, 1, 2);
    });

    return 0;
}
