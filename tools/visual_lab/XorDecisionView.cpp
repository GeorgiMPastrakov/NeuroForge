#include "XorDecisionView.hpp"

#include "imgui.h"
#include "implot.h"

#include <vector>

namespace visual_lab {

void drawXorDecisionView(const neuroforge::DecisionBoundaryGrid& grid, const neuroforge::DatasetSnapshot& dataset) {
    ImGui::TextUnformatted("XOR Decision Map");
    ImGui::SameLine();
    ImGui::TextDisabled("regions + truth-table samples");

    if (!grid.supported || !dataset.supported) {
        ImGui::TextUnformatted(grid.supported ? dataset.message.c_str() : grid.message.c_str());
        return;
    }

    std::vector<double> region_x[2];
    std::vector<double> region_y[2];

    for (size_t row = 0; row < grid.rows; ++row) {
        for (size_t col = 0; col < grid.cols; ++col) {
            const size_t index = row * grid.cols + col;
            const size_t class_index = grid.classes[index] == 0 ? 0 : 1;
            region_x[class_index].push_back(grid.x_values[col]);
            region_y[class_index].push_back(grid.y_values[row]);
        }
    }

    std::vector<double> sample_x[2];
    std::vector<double> sample_y[2];

    for (const neuroforge::DatasetPointSnapshot& point : dataset.points) {
        const size_t class_index = point.class_index == 0 ? 0 : 1;
        sample_x[class_index].push_back(point.x);
        sample_y[class_index].push_back(point.y);
    }

    if (ImPlot::BeginPlot("XOR Classification Regions", ImVec2(-1.0f, 270.0f), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("x0", "x1");
        ImPlot::SetupAxesLimits(-0.30, 1.30, -0.30, 1.30, ImGuiCond_Always);

        ImPlotSpec region_0;
        region_0.Marker = ImPlotMarker_Square;
        region_0.MarkerSize = 3.0f;
        region_0.MarkerFillColor = ImVec4(0.18f, 0.45f, 0.82f, 0.35f);
        region_0.MarkerLineColor = ImVec4(0.18f, 0.45f, 0.82f, 0.35f);
        ImPlot::PlotScatter("region 0", region_x[0].data(), region_y[0].data(), static_cast<int>(region_x[0].size()), region_0);

        ImPlotSpec region_1;
        region_1.Marker = ImPlotMarker_Square;
        region_1.MarkerSize = 3.0f;
        region_1.MarkerFillColor = ImVec4(0.92f, 0.46f, 0.22f, 0.35f);
        region_1.MarkerLineColor = ImVec4(0.92f, 0.46f, 0.22f, 0.35f);
        ImPlot::PlotScatter("region 1", region_x[1].data(), region_y[1].data(), static_cast<int>(region_x[1].size()), region_1);

        ImPlotSpec target_0;
        target_0.Marker = ImPlotMarker_Circle;
        target_0.MarkerSize = 9.0f;
        target_0.MarkerFillColor = ImVec4(0.18f, 0.45f, 0.82f, 1.0f);
        target_0.MarkerLineColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        target_0.LineWeight = 2.0f;
        ImPlot::PlotScatter("target 0", sample_x[0].data(), sample_y[0].data(), static_cast<int>(sample_x[0].size()), target_0);

        ImPlotSpec target_1;
        target_1.Marker = ImPlotMarker_Diamond;
        target_1.MarkerSize = 9.0f;
        target_1.MarkerFillColor = ImVec4(0.92f, 0.46f, 0.22f, 1.0f);
        target_1.MarkerLineColor = ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
        target_1.LineWeight = 2.0f;
        ImPlot::PlotScatter("target 1", sample_x[1].data(), sample_y[1].data(), static_cast<int>(sample_x[1].size()), target_1);
        ImPlot::EndPlot();
    }

    ImGui::TextColored(ImVec4(0.18f, 0.55f, 0.95f, 1.0f), "Blue circles: target 0");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.52f, 0.26f, 1.0f), "Orange diamonds: target 1");
}

}
