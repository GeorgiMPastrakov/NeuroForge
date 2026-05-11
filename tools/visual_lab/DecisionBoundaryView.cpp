#include "DecisionBoundaryView.hpp"

#include "imgui.h"
#include "implot.h"

#include <algorithm>
#include <string>
#include <vector>

namespace visual_lab {

void drawDecisionBoundaryView(const neuroforge::DecisionBoundaryGrid& grid) {
    ImGui::TextUnformatted("Decision Boundary");

    if (!grid.supported) {
        ImGui::TextUnformatted(grid.message.c_str());
        return;
    }

    if (grid.classes.empty()) {
        ImGui::TextUnformatted("No decision data");
        return;
    }

    const size_t class_count = std::max<size_t>(grid.class_count, 1);
    std::vector<std::vector<double>> x(class_count);
    std::vector<std::vector<double>> y(class_count);

    for (size_t row = 0; row < grid.rows; ++row) {
        for (size_t col = 0; col < grid.cols; ++col) {
            const size_t index = row * grid.cols + col;
            const size_t class_index = std::min(grid.classes[index], class_count - 1);
            x[class_index].push_back(grid.x_values[col]);
            y[class_index].push_back(grid.y_values[row]);
        }
    }

    if (ImPlot::BeginPlot("Decision Regions", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("x0", "x1");

        for (size_t index = 0; index < class_count; ++index) {
            if (!x[index].empty()) {
                const std::string label = "predict " + std::to_string(index);
                ImPlot::PlotScatter(label.c_str(), x[index].data(), y[index].data(), static_cast<int>(x[index].size()));
            }
        }

        ImPlot::EndPlot();
    }
}

}
