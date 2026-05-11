#pragma once

#include "neuroforge/neuroforge.hpp"
#include "neuroforge/visualization/VisualSession.hpp"
#include "neuroforge/visualization/VisualizationData.hpp"

#include <array>
#include <vector>

class VisualApp {
public:
    VisualApp();
    int run();

private:
    neuroforge::VisualSession session_;
    neuroforge::ModelSnapshot model_snapshot_;
    neuroforge::NetworkGraphSnapshot network_graph_;
    neuroforge::LossHistorySnapshot loss_snapshot_;
    neuroforge::DatasetSnapshot dataset_snapshot_;
    neuroforge::PredictionSnapshot prediction_snapshot_;
    std::vector<neuroforge::TensorSnapshot> tensor_snapshots_;
    std::vector<neuroforge::GradientSnapshot> gradient_snapshots_;
    neuroforge::DecisionBoundaryGrid decision_grid_;
    std::array<char, 512> model_path_{};
    std::array<char, 512> csv_path_{};
    int label_column_ = 2;
    bool has_header_ = true;
    int class_count_ = 2;
    int epochs_ = 1000;
    float learning_rate_ = 0.1f;
    int batch_size_ = 0;
    bool shuffle_ = false;
    int seed_ = 0;
    int loss_index_ = 0;
    int optimizer_index_ = 0;

    void refreshSnapshots();
    void drawFrame();
    void drawControls();
    void drawPredictionView() const;
    neuroforge::VisualTrainingConfig trainingConfig() const;
};
