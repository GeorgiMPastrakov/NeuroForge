#include "neuroforge/visualization/VisualizationData.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace neuroforge {

ModelSnapshot buildModelSnapshot(Sequential& model) {
    ModelSnapshot snapshot{{}, 0};

    for (const auto& layer : model.layers()) {
        size_t count = 0;

        for (Parameter* parameter : layer->parameters()) {
            count += parameter->data().size();
        }

        snapshot.layers.push_back({layer->name(), count});
        snapshot.total_parameters += count;
    }

    return snapshot;
}

LossHistorySnapshot buildLossHistorySnapshot(const TrainingHistory& history) {
    return {history.epochs(), history.losses()};
}

TensorSnapshot buildTensorSnapshot(const std::string& name, const Tensor& tensor, size_t max_values) {
    if (max_values == 0) {
        throw std::invalid_argument("TensorSnapshot max values must be greater than zero.");
    }

    const size_t count = std::min(max_values, tensor.size());
    std::vector<double> values;
    values.reserve(count);

    for (size_t index = 0; index < count; ++index) {
        values.push_back(tensor.data()[index]);
    }

    return {name, tensor.shape(), std::move(values), tensor.size(), count < tensor.size()};
}

std::vector<GradientSnapshot> buildGradientSnapshots(Sequential& model, size_t max_values) {
    std::vector<GradientSnapshot> snapshots;

    for (Parameter* parameter : model.parameters()) {
        double squared = 0.0;

        for (double value : parameter->grad().data()) {
            squared += value * value;
        }

        TensorSnapshot values = buildTensorSnapshot(parameter->name(), parameter->grad(), max_values);
        snapshots.push_back({parameter->name(), parameter->grad().shape(), std::sqrt(squared), values});
    }

    return snapshots;
}

DecisionBoundaryGrid buildDecisionBoundaryGrid(Sequential& model, double x_min, double x_max, double y_min, double y_max, size_t rows, size_t cols) {
    if (rows < 2 || cols < 2) {
        throw std::invalid_argument("DecisionBoundaryGrid requires at least two rows and columns.");
    }

    DecisionBoundaryGrid grid{rows, cols, {}, {}, {}};
    grid.x_values.reserve(cols);
    grid.y_values.reserve(rows);
    grid.predictions.reserve(rows * cols);

    for (size_t col = 0; col < cols; ++col) {
        const double ratio = static_cast<double>(col) / static_cast<double>(cols - 1);
        grid.x_values.push_back(x_min + ratio * (x_max - x_min));
    }

    for (size_t row = 0; row < rows; ++row) {
        const double ratio = static_cast<double>(row) / static_cast<double>(rows - 1);
        grid.y_values.push_back(y_min + ratio * (y_max - y_min));
    }

    for (double y : grid.y_values) {
        for (double x : grid.x_values) {
            Tensor point = Tensor::fromVector({
                {x, y}
            });
            Tensor output = model.forward(point);
            grid.predictions.push_back(output.data().front());
        }
    }

    return grid;
}

}
