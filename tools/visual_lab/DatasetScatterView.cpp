#include "DatasetScatterView.hpp"

#include "imgui.h"
#include "implot.h"

#include <algorithm>
#include <string>
#include <vector>

namespace visual_lab {

void drawDatasetScatterView(const neuroforge::DatasetSnapshot& snapshot) {
    ImGui::TextUnformatted("Dataset");

    if (!snapshot.supported) {
        ImGui::TextUnformatted(snapshot.message.c_str());
        return;
    }

    if (snapshot.points.empty()) {
        ImGui::TextUnformatted("No dataset loaded");
        return;
    }

    const size_t class_count = std::max<size_t>(snapshot.class_count, 1);
    std::vector<std::vector<double>> x(class_count);
    std::vector<std::vector<double>> y(class_count);

    for (const neuroforge::DatasetPointSnapshot& point : snapshot.points) {
        const size_t index = std::min(point.class_index, class_count - 1);
        x[index].push_back(point.x);
        y[index].push_back(point.y);
    }

    if (ImPlot::BeginPlot("Dataset Scatter", ImVec2(-1, 260))) {
        ImPlot::SetupAxes("x0", "x1");

        for (size_t index = 0; index < class_count; ++index) {
            if (!x[index].empty()) {
                const std::string label = "class " + std::to_string(index);
                ImPlot::PlotScatter(label.c_str(), x[index].data(), y[index].data(), static_cast<int>(x[index].size()));
            }
        }

        ImPlot::EndPlot();
    }
}

}
