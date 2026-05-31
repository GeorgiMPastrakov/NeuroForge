#include "LossPlotView.hpp"

#include "imgui.h"
#include "implot.h"

#include <vector>

namespace visual_lab {

void drawLossPlotView(const neuroforge::LossHistorySnapshot& snapshot) {
    ImGui::TextUnformatted("Training Loss");
    ImGui::SameLine();
    ImGui::TextDisabled("MSE over %zu epochs", snapshot.epochs.size());

    if (snapshot.losses.empty()) {
        ImGui::TextUnformatted("No loss data");
        return;
    }

    ImGui::Text("Final loss: %.6f", snapshot.losses.back());

    std::vector<double> epochs;
    epochs.reserve(snapshot.epochs.size());

    for (size_t epoch : snapshot.epochs) {
        epochs.push_back(static_cast<double>(epoch));
    }

    if (ImPlot::BeginPlot("MSE Convergence", ImVec2(-1.0f, 270.0f), ImPlotFlags_NoLegend)) {
        ImPlot::SetupAxes("Epoch", "MSE");
        ImPlot::PlotLine("MSE", epochs.data(), snapshot.losses.data(), static_cast<int>(snapshot.losses.size()));
        ImPlot::EndPlot();
    }
}

}
