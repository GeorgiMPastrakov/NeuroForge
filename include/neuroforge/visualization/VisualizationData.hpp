#pragma once

#include "neuroforge/core/Shape.hpp"
#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/data/Dataset.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace neuroforge {

struct LayerInfo {
    std::string name;
    size_t parameter_count;
};

struct ModelSnapshot {
    std::vector<LayerInfo> layers;
    size_t total_parameters;
};

struct LossHistorySnapshot {
    std::vector<size_t> epochs;
    std::vector<double> losses;
};

struct TensorSnapshot {
    std::string name;
    Shape shape;
    std::vector<double> values;
    size_t original_size;
    bool truncated;
};

struct GradientSnapshot {
    std::string name;
    Shape shape;
    double norm;
    TensorSnapshot values;
};

struct DecisionBoundaryGrid {
    size_t rows;
    size_t cols;
    bool supported;
    std::string message;
    std::vector<double> x_values;
    std::vector<double> y_values;
    std::vector<double> predictions;
    std::vector<size_t> classes;
    size_t class_count;
};

enum class PredictionKind {
    Empty,
    Regression,
    Binary,
    Multiclass
};

struct PredictionRowSnapshot {
    std::vector<double> input;
    std::vector<double> output;
    std::vector<double> target;
    size_t predicted_class;
    size_t target_class;
};

struct PredictionSnapshot {
    PredictionKind kind;
    bool supported;
    std::string message;
    std::vector<PredictionRowSnapshot> rows;
};

struct DatasetPointSnapshot {
    double x;
    double y;
    double label;
    size_t class_index;
};

struct DatasetSnapshot {
    bool supported;
    std::string message;
    size_t class_count;
    std::vector<DatasetPointSnapshot> points;
};

struct NetworkGraphNode {
    std::string label;
    double x;
    double y;
    size_t layer_index;
    size_t neuron_index;
    bool aggregate;
};

struct NetworkGraphEdge {
    size_t from;
    size_t to;
    double weight;
};

struct NetworkGraphSnapshot {
    bool supported;
    std::string message;
    std::vector<NetworkGraphNode> nodes;
    std::vector<NetworkGraphEdge> edges;
    std::vector<LayerInfo> layers;
};

ModelSnapshot buildModelSnapshot(Sequential& model);
LossHistorySnapshot buildLossHistorySnapshot(const TrainingHistory& history);
TensorSnapshot buildTensorSnapshot(const std::string& name, const Tensor& tensor, size_t max_values);
std::vector<GradientSnapshot> buildGradientSnapshots(Sequential& model, size_t max_values);
DecisionBoundaryGrid buildDecisionBoundaryGrid(Sequential& model, double x_min, double x_max, double y_min, double y_max, size_t rows, size_t cols);
DecisionBoundaryGrid buildDecisionBoundaryGrid(Sequential& model, const Dataset& dataset, double x_min, double x_max, double y_min, double y_max, size_t rows, size_t cols);
DatasetSnapshot buildDatasetSnapshot(const Dataset& dataset);
PredictionSnapshot buildPredictionSnapshot(const Dataset* dataset, const Tensor* prediction, size_t class_count);
NetworkGraphSnapshot buildNetworkGraphSnapshot(Sequential& model, size_t max_neurons_per_layer);

}
