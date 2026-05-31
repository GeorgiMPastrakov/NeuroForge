#pragma once

#include "neuroforge/core/Tensor.hpp"
#include "neuroforge/data/Dataset.hpp"
#include "neuroforge/nn/Sequential.hpp"
#include "neuroforge/training/TrainingHistory.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace neuroforge {

enum class VisualLossType {
    MSE,
    BinaryCrossEntropy,
    CrossEntropy
};

enum class VisualOptimizerType {
    SGD,
    Adam
};

struct VisualTrainingConfig {
    size_t epochs = 1000;
    double learning_rate = 0.1;
    size_t batch_size = 0;
    bool shuffle = false;
    uint32_t seed = 0;
    VisualLossType loss = VisualLossType::MSE;
    VisualOptimizerType optimizer = VisualOptimizerType::SGD;
    size_t class_count = 2;
};

class VisualSession {
public:
    VisualSession();

    bool resetXorDemo();
    bool loadModel(const std::string& path);
    bool loadCsvDataset(const std::string& path, size_t label_column, bool has_header);
    bool trainCurrent(const VisualTrainingConfig& config);
    void refreshPredictions();

    Sequential& model();
    const Sequential& model() const;
    const std::optional<Dataset>& dataset() const;
    const TrainingHistory& history() const;
    const std::optional<Tensor>& predictions() const;
    const std::string& status() const;
    bool hasModel() const;
    bool hasDataset() const;

private:
    Sequential model_;
    std::optional<Dataset> dataset_;
    TrainingHistory history_;
    std::optional<Tensor> predictions_;
    std::string status_;

    Dataset trainingDatasetFor(const VisualTrainingConfig& config) const;
    Tensor oneHotLabels(size_t class_count) const;
    void setError(const std::exception& exception);
};

}
