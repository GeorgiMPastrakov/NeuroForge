#include "GradientView.hpp"

#include "imgui.h"

namespace visual_lab {

void drawGradientView(const std::vector<neuroforge::GradientSnapshot>& gradients) {
    ImGui::TextUnformatted("Gradients");

    if (gradients.empty()) {
        ImGui::TextUnformatted("No gradients");
        return;
    }

    for (const neuroforge::GradientSnapshot& gradient : gradients) {
        ImGui::Separator();
        ImGui::Text("%s %s", gradient.name.c_str(), gradient.shape.toString().c_str());
        ImGui::Text("Norm: %.6f", gradient.norm);
    }
}

}
