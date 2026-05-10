#pragma once

#include "neuroforge/neuroforge.hpp"
#include "neuroforge/visualization/VisualizationData.hpp"

#include <memory>

class VisualApp {
public:
    VisualApp();
    int run();

private:
    neuroforge::Sequential model_;
    neuroforge::Dataset dataset_;
    neuroforge::TrainingHistory history_;
    neuroforge::ModelSnapshot model_snapshot_;
    neuroforge::LossHistorySnapshot loss_snapshot_;
    neuroforge::Tensor prediction_;

    void buildDemoState();
    void refreshSnapshots();
    void drawFrame();
};
