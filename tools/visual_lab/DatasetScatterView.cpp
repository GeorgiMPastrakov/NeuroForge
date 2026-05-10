#include "DatasetScatterView.hpp"

#include "imgui.h"
#include "implot.h"

#include <vector>

namespace visual_lab {

void drawDatasetScatterView(const neuroforge::Dataset& dataset) {
    ImGui::TextUnformatted("Dataset");

    if (dataset.features().shape().cols() != 2 || dataset.labels().shape().cols() != 1) {
        ImGui::TextUnformatted("Scatter view requires 2D features and single labels");
        return;
    }

    std::vector<double> x0;
    std::vector<double> y0;
    std::vector<double> x1;
    std::vector<double> y1;

    for (size_t row = 0; row < dataset.size(); ++row) {
        if (dataset.labels().at(row, 0) >= 0.5) {
            x1.push_back(dataset.features().at(row, 0));
            y1.push_back(dataset.features().at(row, 1));
        } else {
            x0.push_back(dataset.features().at(row, 0));
            y0.push_back(dataset.features().at(row, 1));
        }
    }

    if (ImPlot::BeginPlot("Dataset Scatter", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("x0", "x1");
        if (!x0.empty()) {
            ImPlot::PlotScatter("class 0", x0.data(), y0.data(), static_cast<int>(x0.size()));
        }
        if (!x1.empty()) {
            ImPlot::PlotScatter("class 1", x1.data(), y1.data(), static_cast<int>(x1.size()));
        }
        ImPlot::EndPlot();
    }
}

}
