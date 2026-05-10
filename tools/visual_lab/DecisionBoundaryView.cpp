#include "DecisionBoundaryView.hpp"

#include "imgui.h"
#include "implot.h"

#include <vector>

namespace visual_lab {

void drawDecisionBoundaryView(const neuroforge::DecisionBoundaryGrid& grid) {
    ImGui::TextUnformatted("Decision Boundary");

    if (grid.predictions.empty()) {
        ImGui::TextUnformatted("No decision data");
        return;
    }

    std::vector<double> low_x;
    std::vector<double> low_y;
    std::vector<double> high_x;
    std::vector<double> high_y;

    for (size_t row = 0; row < grid.rows; ++row) {
        for (size_t col = 0; col < grid.cols; ++col) {
            const size_t index = row * grid.cols + col;
            if (grid.predictions[index] >= 0.5) {
                high_x.push_back(grid.x_values[col]);
                high_y.push_back(grid.y_values[row]);
            } else {
                low_x.push_back(grid.x_values[col]);
                low_y.push_back(grid.y_values[row]);
            }
        }
    }

    if (ImPlot::BeginPlot("Decision Regions", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("x0", "x1");
        if (!low_x.empty()) {
            ImPlot::PlotScatter("predict 0", low_x.data(), low_y.data(), static_cast<int>(low_x.size()));
        }
        if (!high_x.empty()) {
            ImPlot::PlotScatter("predict 1", high_x.data(), high_y.data(), static_cast<int>(high_x.size()));
        }
        ImPlot::EndPlot();
    }
}

}
