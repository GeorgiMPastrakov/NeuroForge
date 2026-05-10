#include "ModelGraphView.hpp"

#include "imgui.h"

namespace visual_lab {

void drawModelGraphView(const neuroforge::ModelSnapshot& snapshot) {
    ImGui::TextUnformatted("Model Architecture");
    ImGui::Text("Layers: %zu", snapshot.layers.size());
    ImGui::Text("Parameters: %zu", snapshot.total_parameters);

    if (snapshot.layers.empty()) {
        ImGui::TextUnformatted("No layers");
        return;
    }

    for (size_t index = 0; index < snapshot.layers.size(); ++index) {
        const neuroforge::LayerInfo& layer = snapshot.layers[index];
        ImGui::Separator();
        ImGui::Text("[%zu] %s", index, layer.name.c_str());
        ImGui::Text("Parameters: %zu", layer.parameter_count);
    }
}

}
