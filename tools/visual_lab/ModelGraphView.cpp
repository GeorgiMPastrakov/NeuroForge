#include "ModelGraphView.hpp"

#include "imgui.h"

#include <algorithm>
#include <cmath>

namespace visual_lab {

namespace {

ImU32 edgeColor(double weight) {
    const float alpha = static_cast<float>(std::clamp(std::fabs(weight), 0.15, 1.0));

    if (weight >= 0.0) {
        return ImGui::ColorConvertFloat4ToU32(ImVec4(0.20f, 0.75f, 0.45f, alpha));
    }

    return ImGui::ColorConvertFloat4ToU32(ImVec4(0.95f, 0.35f, 0.30f, alpha));
}

float edgeWidth(double weight) {
    return 1.0f + static_cast<float>(std::clamp(std::fabs(weight), 0.0, 2.0));
}

}

void drawModelGraphView(const neuroforge::NetworkGraphSnapshot& graph, const neuroforge::ModelSnapshot& snapshot) {
    ImGui::TextUnformatted("Model Architecture");
    ImGui::Text("Layers: %zu", snapshot.layers.size());
    ImGui::Text("Parameters: %zu", snapshot.total_parameters);

    if (!graph.supported) {
        ImGui::TextUnformatted(graph.message.c_str());
    } else if (graph.nodes.empty()) {
        ImGui::TextUnformatted("No graph data");
    } else {
        const ImVec2 origin = ImGui::GetCursorScreenPos();
        const float width = ImGui::GetContentRegionAvail().x;
        const float height = 280.0f;
        const float left = origin.x + 30.0f;
        const float top = origin.y + 24.0f;
        const float draw_width = std::max(width - 60.0f, 120.0f);
        const float draw_height = height - 54.0f;
        const double max_x = std::max(1.0, graph.nodes.back().x);
        ImDrawList* draw_list = ImGui::GetWindowDrawList();

        for (const neuroforge::NetworkGraphEdge& edge : graph.edges) {
            if (edge.from >= graph.nodes.size() || edge.to >= graph.nodes.size()) {
                continue;
            }

            const neuroforge::NetworkGraphNode& from = graph.nodes[edge.from];
            const neuroforge::NetworkGraphNode& to = graph.nodes[edge.to];
            const ImVec2 from_pos(left + static_cast<float>(from.x / max_x) * draw_width, top + static_cast<float>(from.y) * draw_height);
            const ImVec2 to_pos(left + static_cast<float>(to.x / max_x) * draw_width, top + static_cast<float>(to.y) * draw_height);
            draw_list->AddLine(from_pos, to_pos, edgeColor(edge.weight), edgeWidth(edge.weight));
        }

        for (const neuroforge::NetworkGraphNode& node : graph.nodes) {
            const ImVec2 position(left + static_cast<float>(node.x / max_x) * draw_width, top + static_cast<float>(node.y) * draw_height);
            const float radius = node.aggregate ? 18.0f : 11.0f;
            const ImU32 fill = node.aggregate
                ? ImGui::ColorConvertFloat4ToU32(ImVec4(0.22f, 0.32f, 0.52f, 1.0f))
                : ImGui::ColorConvertFloat4ToU32(ImVec4(0.18f, 0.50f, 0.85f, 1.0f));
            draw_list->AddCircleFilled(position, radius, fill, 24);
            draw_list->AddCircle(position, radius, ImGui::ColorConvertFloat4ToU32(ImVec4(0.90f, 0.94f, 1.0f, 1.0f)), 24, 1.4f);
            draw_list->AddText(ImVec2(position.x + radius + 4.0f, position.y - 7.0f), ImGui::GetColorU32(ImGuiCol_Text), node.label.c_str());
        }

        ImGui::Dummy(ImVec2(width, height));
    }

    if (snapshot.layers.empty()) {
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
