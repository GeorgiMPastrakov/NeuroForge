#pragma once

#include "neuroforge/neuroforge.hpp"
#include "neuroforge/visualization/VisualizationData.hpp"

#include <memory>
#include <vector>

class VisualApp {
public:
    VisualApp();
    int run();

private:
    neuroforge::Sequential model_;
    neuroforge::Dataset dataset_;
    neuroforge::TrainingHistory history_;
    neuroforge::ModelSnapshot model_snapshot_;
    neuroforge::NetworkGraphSnapshot network_graph_;
    neuroforge::LossHistorySnapshot loss_snapshot_;
    neuroforge::Tensor prediction_;
    std::vector<neuroforge::TensorSnapshot> tensor_snapshots_;
    std::vector<neuroforge::GradientSnapshot> gradient_snapshots_;
    neuroforge::DecisionBoundaryGrid decision_grid_;

    void buildDemoState();
    void refreshSnapshots();
    void drawFrame();
};
