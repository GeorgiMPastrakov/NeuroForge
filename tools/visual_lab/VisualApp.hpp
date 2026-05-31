#pragma once

#include "neuroforge/visualization/VisualSession.hpp"
#include "neuroforge/visualization/VisualizationData.hpp"

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
    neuroforge::DecisionBoundaryGrid decision_grid_;

    void refreshSnapshots();
    void drawFrame() const;
    void drawHeader() const;
    void drawPredictionTable() const;
};
