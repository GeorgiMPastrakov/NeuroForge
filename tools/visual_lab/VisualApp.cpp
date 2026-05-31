#include "VisualApp.hpp"

#include "LossPlotView.hpp"
#include "ModelGraphView.hpp"
#include "XorDecisionView.hpp"
#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>

namespace {

neuroforge::DecisionBoundaryGrid emptyDecisionGrid() {
    return {2, 2, false, "No decision data.", {}, {}, {}, {}, 2};
}

}

VisualApp::VisualApp()
    : dataset_snapshot_(neuroforge::DatasetSnapshot{false, "No dataset loaded.", 0, {}}),
      prediction_snapshot_(neuroforge::PredictionSnapshot{neuroforge::PredictionKind::Empty, false, "No prediction data.", {}}),
      decision_grid_(emptyDecisionGrid()) {
    refreshSnapshots();
}

int VisualApp::run() {
    if (!glfwInit()) {
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1440, 900, "NeuroForge Visual Lab", nullptr, nullptr);

    if (window == nullptr) {
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        drawFrame();

        ImGui::Render();
        int display_width = 0;
        int display_height = 0;
        glfwGetFramebufferSize(window, &display_width, &display_height);
        glViewport(0, 0, display_width, display_height);
        glClearColor(0.06f, 0.07f, 0.09f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

void VisualApp::refreshSnapshots() {
    model_snapshot_ = neuroforge::buildModelSnapshot(session_.model());
    network_graph_ = neuroforge::buildNetworkGraphSnapshot(session_.model(), 12);
    loss_snapshot_ = neuroforge::buildLossHistorySnapshot(session_.history());

    if (!session_.dataset().has_value()) {
        return;
    }

    const neuroforge::Dataset& dataset = session_.dataset().value();
    dataset_snapshot_ = neuroforge::buildDatasetSnapshot(dataset);
    const neuroforge::Tensor* predictions = session_.predictions().has_value() ? &session_.predictions().value() : nullptr;
    prediction_snapshot_ = neuroforge::buildPredictionSnapshot(&dataset, predictions, 2);

    const bool was_training = session_.model().isTraining();
    session_.model().eval();
    decision_grid_ = neuroforge::buildDecisionBoundaryGrid(session_.model(), dataset, -0.25, 1.25, -0.25, 1.25, 36, 36);

    if (was_training) {
        session_.model().train();
    }
}

void VisualApp::drawFrame() const {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("NeuroForge Visual Lab", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);

    drawHeader();

    if (ImGui::BeginTable("main_row", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Network", ImGuiTableColumnFlags_WidthStretch, 1.15f);
        ImGui::TableSetupColumn("Loss", ImGuiTableColumnFlags_WidthStretch, 1.0f);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::BeginChild("network_panel", ImVec2(0.0f, 340.0f), true);
        visual_lab::drawModelGraphView(network_graph_, model_snapshot_);
        ImGui::EndChild();
        ImGui::TableSetColumnIndex(1);
        ImGui::BeginChild("loss_panel", ImVec2(0.0f, 340.0f), true);
        visual_lab::drawLossPlotView(loss_snapshot_);
        ImGui::EndChild();
        ImGui::EndTable();
    }

    ImGui::Spacing();

    if (ImGui::BeginTable("bottom_row", 2, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("Decision Map", ImGuiTableColumnFlags_WidthStretch, 1.35f);
        ImGui::TableSetupColumn("Predictions", ImGuiTableColumnFlags_WidthStretch, 0.8f);
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::BeginChild("decision_panel", ImVec2(0.0f, 330.0f), true);
        visual_lab::drawXorDecisionView(decision_grid_, dataset_snapshot_);
        ImGui::EndChild();
        ImGui::TableSetColumnIndex(1);
        ImGui::BeginChild("prediction_panel", ImVec2(0.0f, 330.0f), true);
        drawPredictionTable();
        ImGui::EndChild();
        ImGui::EndTable();
    }

    ImGui::End();
}

void VisualApp::drawHeader() const {
    ImGui::TextUnformatted("NeuroForge Visual Lab");
    ImGui::SameLine();
    ImGui::TextDisabled("XOR Classification Demo");
    ImGui::Separator();

    if (ImGui::BeginTable("summary", 4, ImGuiTableFlags_BordersInnerV | ImGuiTableFlags_SizingStretchSame)) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::TextDisabled("Architecture");
        ImGui::TextUnformatted("2 inputs -> 4 hidden -> 1 output");
        ImGui::TableSetColumnIndex(1);
        ImGui::TextDisabled("Layers");
        ImGui::TextUnformatted("Linear -> Sigmoid -> Linear -> Sigmoid");
        ImGui::TableSetColumnIndex(2);
        ImGui::TextDisabled("Training");
        ImGui::Text("%zu epochs, MSE + SGD", loss_snapshot_.epochs.size());
        ImGui::TableSetColumnIndex(3);
        ImGui::TextDisabled("Result");
        ImGui::Text("%zu parameters, loss %.6f", model_snapshot_.total_parameters, loss_snapshot_.losses.empty() ? 0.0 : loss_snapshot_.losses.back());
        ImGui::EndTable();
    }

    ImGui::Spacing();
}

void VisualApp::drawPredictionTable() const {
    ImGui::TextUnformatted("XOR Predictions");
    ImGui::TextDisabled("The model should reproduce the XOR truth table.");
    ImGui::Spacing();

    if (!prediction_snapshot_.supported || prediction_snapshot_.rows.empty()) {
        ImGui::TextUnformatted(prediction_snapshot_.message.c_str());
        return;
    }

    if (!ImGui::BeginTable("predictions", 5, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchSame)) {
        return;
    }

    ImGui::TableSetupColumn("x0");
    ImGui::TableSetupColumn("x1");
    ImGui::TableSetupColumn("Target");
    ImGui::TableSetupColumn("Output");
    ImGui::TableSetupColumn("Class");
    ImGui::TableHeadersRow();

    for (const neuroforge::PredictionRowSnapshot& row : prediction_snapshot_.rows) {
        ImGui::TableNextRow();
        ImGui::TableSetColumnIndex(0);
        ImGui::Text("%.0f", row.input[0]);
        ImGui::TableSetColumnIndex(1);
        ImGui::Text("%.0f", row.input[1]);
        ImGui::TableSetColumnIndex(2);
        ImGui::Text("%.0f", row.target[0]);
        ImGui::TableSetColumnIndex(3);
        ImGui::Text("%.4f", row.output[0]);
        ImGui::TableSetColumnIndex(4);
        ImGui::TextColored(row.predicted_class == row.target_class ? ImVec4(0.35f, 0.90f, 0.55f, 1.0f) : ImVec4(0.95f, 0.35f, 0.30f, 1.0f), "%zu", row.predicted_class);
    }

    ImGui::EndTable();
    ImGui::Spacing();
    ImGui::TextDisabled("Green class values are correct predictions.");
}
