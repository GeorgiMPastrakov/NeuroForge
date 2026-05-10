#include "TensorInspectorView.hpp"

#include "imgui.h"

namespace visual_lab {

void drawTensorInspectorView(const std::vector<neuroforge::TensorSnapshot>& tensors) {
    ImGui::TextUnformatted("Tensors");

    if (tensors.empty()) {
        ImGui::TextUnformatted("No tensors");
        return;
    }

    for (const neuroforge::TensorSnapshot& tensor : tensors) {
        ImGui::Separator();
        ImGui::Text("%s %s", tensor.name.c_str(), tensor.shape.toString().c_str());

        for (size_t index = 0; index < tensor.values.size(); ++index) {
            ImGui::Text("[%zu] %.6f", index, tensor.values[index]);
        }

        if (tensor.truncated) {
            ImGui::Text("Showing %zu of %zu", tensor.values.size(), tensor.original_size);
        }
    }
}

}
