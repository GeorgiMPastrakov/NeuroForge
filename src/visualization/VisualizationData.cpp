#include "neuroforge/visualization/VisualizationData.hpp"

#include "neuroforge/nn/Dropout.hpp"
#include "neuroforge/nn/LeakyReLU.hpp"
#include "neuroforge/nn/Linear.hpp"
#include "neuroforge/nn/ReLU.hpp"
#include "neuroforge/nn/Sigmoid.hpp"
#include "neuroforge/nn/Softmax.hpp"
#include "neuroforge/nn/Tanh.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace neuroforge {

namespace {

size_t classIndexFromRow(const Tensor& tensor, size_t row) {
    if (tensor.shape().cols() == 1) {
        const double value = tensor.at(row, 0);
        const auto rounded = static_cast<long long>(std::llround(value));

        if (rounded < 0 || std::fabs(value - static_cast<double>(rounded)) > 1e-9) {
            return value >= 0.5 ? 1 : 0;
        }

        return static_cast<size_t>(rounded);
    }

    size_t best = 0;
    double best_value = tensor.at(row, 0);

    for (size_t col = 1; col < tensor.shape().cols(); ++col) {
        if (tensor.at(row, col) > best_value) {
            best = col;
            best_value = tensor.at(row, col);
        }
    }

    return best;
}

size_t outputClass(const Tensor& output, size_t row) {
    if (output.shape().cols() == 1) {
        return output.at(row, 0) >= 0.5 ? 1 : 0;
    }

    Tensor probabilities = output.softmaxRows();
    return classIndexFromRow(probabilities, row);
}

std::vector<double> rowValues(const Tensor& tensor, size_t row) {
    std::vector<double> values;
    values.reserve(tensor.shape().cols());

    for (size_t col = 0; col < tensor.shape().cols(); ++col) {
        values.push_back(tensor.at(row, col));
    }

    return values;
}

bool isActivationLayer(const Module& module) {
    return dynamic_cast<const ReLU*>(&module) != nullptr
        || dynamic_cast<const Sigmoid*>(&module) != nullptr
        || dynamic_cast<const Tanh*>(&module) != nullptr
        || dynamic_cast<const LeakyReLU*>(&module) != nullptr
        || dynamic_cast<const Softmax*>(&module) != nullptr
        || dynamic_cast<const Dropout*>(&module) != nullptr;
}

void fillBoundaryAxes(DecisionBoundaryGrid& grid, double x_min, double x_max, double y_min, double y_max) {
    grid.x_values.reserve(grid.cols);
    grid.y_values.reserve(grid.rows);

    for (size_t col = 0; col < grid.cols; ++col) {
        const double ratio = static_cast<double>(col) / static_cast<double>(grid.cols - 1);
        grid.x_values.push_back(x_min + ratio * (x_max - x_min));
    }

    for (size_t row = 0; row < grid.rows; ++row) {
        const double ratio = static_cast<double>(row) / static_cast<double>(grid.rows - 1);
        grid.y_values.push_back(y_min + ratio * (y_max - y_min));
    }
}

}

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

    DecisionBoundaryGrid grid{rows, cols, true, "", {}, {}, {}, {}, 2};
    fillBoundaryAxes(grid, x_min, x_max, y_min, y_max);
    grid.predictions.reserve(rows * cols);
    grid.classes.reserve(rows * cols);

    for (size_t row = 0; row < grid.y_values.size(); ++row) {
        for (size_t col = 0; col < grid.x_values.size(); ++col) {
            Tensor point = Tensor::fromVector({
                {grid.x_values[col], grid.y_values[row]}
            });
            Tensor output = model.forward(point);
            const size_t predicted_class = outputClass(output, 0);
            grid.predictions.push_back(output.shape().cols() == 1 ? output.at(0, 0) : static_cast<double>(predicted_class));
            grid.classes.push_back(predicted_class);
            grid.class_count = std::max(grid.class_count, predicted_class + 1);
        }
    }

    return grid;
}

DecisionBoundaryGrid buildDecisionBoundaryGrid(Sequential& model, const Dataset& dataset, double x_min, double x_max, double y_min, double y_max, size_t rows, size_t cols) {
    if (rows < 2 || cols < 2) {
        throw std::invalid_argument("DecisionBoundaryGrid requires at least two rows and columns.");
    }

    DecisionBoundaryGrid grid{rows, cols, true, "", {}, {}, {}, {}, 2};
    fillBoundaryAxes(grid, x_min, x_max, y_min, y_max);

    if (dataset.features().shape().cols() != 2) {
        grid.supported = false;
        grid.message = "Decision boundary requires 2D features.";
        return grid;
    }

    grid.predictions.reserve(rows * cols);
    grid.classes.reserve(rows * cols);

    for (size_t row = 0; row < grid.y_values.size(); ++row) {
        for (size_t col = 0; col < grid.x_values.size(); ++col) {
            Tensor point = Tensor::fromVector({
                {grid.x_values[col], grid.y_values[row]}
            });
            Tensor output = model.forward(point);
            const size_t predicted_class = outputClass(output, 0);
            grid.predictions.push_back(output.shape().cols() == 1 ? output.at(0, 0) : static_cast<double>(predicted_class));
            grid.classes.push_back(predicted_class);
            grid.class_count = std::max(grid.class_count, predicted_class + 1);
        }
    }

    return grid;
}

DatasetSnapshot buildDatasetSnapshot(const Dataset& dataset) {
    DatasetSnapshot snapshot{true, "", 0, {}};

    if (dataset.features().shape().cols() != 2) {
        snapshot.supported = false;
        snapshot.message = "Dataset scatter requires 2D features.";
        return snapshot;
    }

    snapshot.points.reserve(dataset.size());

    for (size_t row = 0; row < dataset.size(); ++row) {
        const size_t class_index = classIndexFromRow(dataset.labels(), row);
        const double label = dataset.labels().shape().cols() == 1 ? dataset.labels().at(row, 0) : static_cast<double>(class_index);
        snapshot.points.push_back({dataset.features().at(row, 0), dataset.features().at(row, 1), label, class_index});
        snapshot.class_count = std::max(snapshot.class_count, class_index + 1);
    }

    return snapshot;
}

PredictionSnapshot buildPredictionSnapshot(const Dataset* dataset, const Tensor* prediction, size_t class_count) {
    if (dataset == nullptr || prediction == nullptr) {
        return {PredictionKind::Empty, false, "No prediction data.", {}};
    }

    if (prediction->rank() != 2 || prediction->shape().rows() != dataset->size()) {
        return {PredictionKind::Empty, false, "Prediction shape does not match dataset rows.", {}};
    }

    PredictionKind kind = PredictionKind::Regression;

    if (prediction->shape().cols() > 1) {
        kind = PredictionKind::Multiclass;
    } else if (class_count <= 2 && dataset->labels().shape().cols() == 1) {
        kind = PredictionKind::Binary;
    }

    PredictionSnapshot snapshot{kind, true, "", {}};
    snapshot.rows.reserve(dataset->size());

    for (size_t row = 0; row < dataset->size(); ++row) {
        snapshot.rows.push_back({
            rowValues(dataset->features(), row),
            rowValues(*prediction, row),
            rowValues(dataset->labels(), row),
            outputClass(*prediction, row),
            classIndexFromRow(dataset->labels(), row)
        });
    }

    return snapshot;
}

NetworkGraphSnapshot buildNetworkGraphSnapshot(Sequential& model, size_t max_neurons_per_layer) {
    if (max_neurons_per_layer == 0) {
        throw std::invalid_argument("Network graph max neurons must be greater than zero.");
    }

    NetworkGraphSnapshot snapshot{true, "", {}, {}, {}};
    snapshot.layers = buildModelSnapshot(model).layers;

    if (model.size() == 0) {
        snapshot.supported = false;
        snapshot.message = "No model loaded.";
        return snapshot;
    }

    std::vector<std::vector<size_t>> dense_node_indices;
    size_t dense_layer_index = 0;

    for (const auto& module : model.layers()) {
        if (const auto* linear = dynamic_cast<const Linear*>(module.get())) {
            if (dense_node_indices.empty()) {
                std::vector<size_t> input_nodes;

                if (linear->inFeatures() > max_neurons_per_layer) {
                    snapshot.nodes.push_back({"input x" + std::to_string(linear->inFeatures()), 0.0, 0.5, 0, 0, true});
                    input_nodes.push_back(snapshot.nodes.size() - 1);
                } else {
                    for (size_t index = 0; index < linear->inFeatures(); ++index) {
                        const double y = linear->inFeatures() == 1 ? 0.5 : static_cast<double>(index) / static_cast<double>(linear->inFeatures() - 1);
                        snapshot.nodes.push_back({"x" + std::to_string(index), 0.0, y, 0, index, false});
                        input_nodes.push_back(snapshot.nodes.size() - 1);
                    }
                }

                dense_node_indices.push_back(std::move(input_nodes));
            }

            std::vector<size_t> output_nodes;
            const double x = static_cast<double>(dense_layer_index + 1);

            if (linear->outFeatures() > max_neurons_per_layer) {
                snapshot.nodes.push_back({"Linear x" + std::to_string(linear->outFeatures()), x, 0.5, dense_layer_index + 1, 0, true});
                output_nodes.push_back(snapshot.nodes.size() - 1);
            } else {
                for (size_t output = 0; output < linear->outFeatures(); ++output) {
                    const double y = linear->outFeatures() == 1 ? 0.5 : static_cast<double>(output) / static_cast<double>(linear->outFeatures() - 1);
                    snapshot.nodes.push_back({"n" + std::to_string(output), x, y, dense_layer_index + 1, output, false});
                    output_nodes.push_back(snapshot.nodes.size() - 1);
                }
            }

            const std::vector<size_t>& input_nodes = dense_node_indices.back();

            for (size_t input_node_index = 0; input_node_index < input_nodes.size(); ++input_node_index) {
                for (size_t output_node_index = 0; output_node_index < output_nodes.size(); ++output_node_index) {
                    double weight = 0.0;

                    if (!snapshot.nodes[input_nodes[input_node_index]].aggregate && !snapshot.nodes[output_nodes[output_node_index]].aggregate) {
                        weight = linear->weights().data().at(input_node_index, output_node_index);
                    }

                    snapshot.edges.push_back({input_nodes[input_node_index], output_nodes[output_node_index], weight});
                }
            }

            dense_node_indices.push_back(std::move(output_nodes));
            ++dense_layer_index;
        } else if (!isActivationLayer(*module)) {
            snapshot.supported = false;
            snapshot.message = "Network graph supports dense Sequential layers only.";
            return snapshot;
        }
    }

    if (dense_node_indices.size() < 2) {
        snapshot.supported = false;
        snapshot.message = "Network graph requires at least one Linear layer.";
    }

    return snapshot;
}

}
