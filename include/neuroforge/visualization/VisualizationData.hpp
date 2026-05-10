#pragma once

#include "neuroforge/core/Shape.hpp"
#include "neuroforge/core/Tensor.hpp"
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
    std::vector<double> x_values;
    std::vector<double> y_values;
    std::vector<double> predictions;
};

ModelSnapshot buildModelSnapshot(Sequential& model);
LossHistorySnapshot buildLossHistorySnapshot(const TrainingHistory& history);
TensorSnapshot buildTensorSnapshot(const std::string& name, const Tensor& tensor, size_t max_values);
std::vector<GradientSnapshot> buildGradientSnapshots(Sequential& model, size_t max_values);
DecisionBoundaryGrid buildDecisionBoundaryGrid(Sequential& model, double x_min, double x_max, double y_min, double y_max, size_t rows, size_t cols);

}
