#include "ModelGraphView.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace visual_lab {

namespace {

ImU32 edgeColor(double weight) {
    const float alpha = static_cast<float>(std::clamp(std::fabs(weight), 0.18, 0.92));
    return weight >= 0.0
        ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.20f, 0.78f, 0.48f, alpha))
        : ImGui::ColorConvertFloat4ToU32(ImVec4(0.95f, 0.34f, 0.30f, alpha));
}

float edgeWidth(double weight) {
    return 1.0f + static_cast<float>(std::clamp(std::fabs(weight), 0.0, 2.0));
}

}

void drawModelGraphView(const neuroforge::NetworkGraphSnapshot& graph, const neuroforge::ModelSnapshot& snapshot) {
    ImGui::TextUnformatted("Network Architecture");
    ImGui::SameLine();
    ImGui::TextDisabled("%zu parameters", snapshot.total_parameters);

    if (!graph.supported || graph.nodes.empty()) {
        ImGui::TextUnformatted(graph.message.c_str());
        return;
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    const float width = ImGui::GetContentRegionAvail().x;
    const float height = 250.0f;
    const float left = origin.x + 48.0f;
    const float top = origin.y + 24.0f;
    const float draw_width = std::max(width - 116.0f, 160.0f);
    const float draw_height = height - 54.0f;
    const double max_x = std::max(1.0, graph.nodes.back().x);
    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    for (const neuroforge::NetworkGraphEdge& edge : graph.edges) {
        const neuroforge::NetworkGraphNode& from = graph.nodes[edge.from];
        const neuroforge::NetworkGraphNode& to = graph.nodes[edge.to];
        const ImVec2 from_pos(left + static_cast<float>(from.x / max_x) * draw_width, top + static_cast<float>(from.y) * draw_height);
        const ImVec2 to_pos(left + static_cast<float>(to.x / max_x) * draw_width, top + static_cast<float>(to.y) * draw_height);
        draw_list->AddLine(from_pos, to_pos, edgeColor(edge.weight), edgeWidth(edge.weight));
    }

    for (const neuroforge::NetworkGraphNode& node : graph.nodes) {
        const ImVec2 position(left + static_cast<float>(node.x / max_x) * draw_width, top + static_cast<float>(node.y) * draw_height);
        draw_list->AddCircleFilled(position, 11.0f, ImGui::ColorConvertFloat4ToU32(ImVec4(0.16f, 0.52f, 0.88f, 1.0f)), 24);
        draw_list->AddCircle(position, 11.0f, ImGui::ColorConvertFloat4ToU32(ImVec4(0.90f, 0.94f, 1.0f, 1.0f)), 24, 1.4f);
        draw_list->AddText(ImVec2(position.x + 16.0f, position.y - 7.0f), ImGui::GetColorU32(ImGuiCol_Text), node.label.c_str());
    }

    ImGui::Dummy(ImVec2(width, height));
    ImGui::TextColored(ImVec4(0.20f, 0.78f, 0.48f, 1.0f), "Positive weight");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.95f, 0.34f, 0.30f, 1.0f), "Negative weight");
    ImGui::TextDisabled("Thicker connection = larger weight magnitude");
}

}
