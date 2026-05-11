#include "VisualApp.hpp"

#include "DatasetScatterView.hpp"
#include "DecisionBoundaryView.hpp"
#include "GradientView.hpp"
#include "LossPlotView.hpp"
#include "ModelGraphView.hpp"
#include "TensorInspectorView.hpp"
#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

namespace {

neuroforge::DecisionBoundaryGrid emptyDecisionGrid(const std::string& message) {
    return {2, 2, false, message, {}, {}, {}, {}, 2};
}

std::pair<double, double> paddedRange(double min_value, double max_value) {
    if (min_value == max_value) {
        return {min_value - 1.0, max_value + 1.0};
    }

    const double padding = (max_value - min_value) * 0.15;
    return {min_value - padding, max_value + padding};
}

}

VisualApp::VisualApp()
    : dataset_snapshot_(neuroforge::DatasetSnapshot{false, "No dataset loaded.", 0, {}}),
      prediction_snapshot_(neuroforge::PredictionSnapshot{neuroforge::PredictionKind::Empty, false, "No prediction data.", {}}),
      decision_grid_(emptyDecisionGrid("No decision data.")) {
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
        glClearColor(0.08f, 0.09f, 0.10f, 1.0f);
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
    gradient_snapshots_ = neuroforge::buildGradientSnapshots(session_.model(), 12);
    tensor_snapshots_.clear();

    for (neuroforge::Parameter* parameter : session_.model().parameters()) {
        tensor_snapshots_.push_back(neuroforge::buildTensorSnapshot(parameter->name(), parameter->data(), 12));
    }

    const neuroforge::Dataset* dataset = session_.dataset().has_value() ? &session_.dataset().value() : nullptr;
    const neuroforge::Tensor* predictions = session_.predictions().has_value() ? &session_.predictions().value() : nullptr;
    prediction_snapshot_ = neuroforge::buildPredictionSnapshot(dataset, predictions, static_cast<size_t>(std::max(class_count_, 1)));

    if (dataset == nullptr) {
        dataset_snapshot_ = neuroforge::DatasetSnapshot{false, "No dataset loaded.", 0, {}};
        decision_grid_ = emptyDecisionGrid("No dataset loaded.");
        return;
    }

    dataset_snapshot_ = neuroforge::buildDatasetSnapshot(*dataset);

    if (!session_.hasModel()) {
        decision_grid_ = emptyDecisionGrid("No model loaded.");
        return;
    }

    if (dataset->features().shape().cols() != 2) {
        decision_grid_ = neuroforge::buildDecisionBoundaryGrid(session_.model(), *dataset, 0.0, 1.0, 0.0, 1.0, 30, 30);
        return;
    }

    double min_x = dataset->features().at(0, 0);
    double max_x = min_x;
    double min_y = dataset->features().at(0, 1);
    double max_y = min_y;

    for (size_t row = 1; row < dataset->size(); ++row) {
        min_x = std::min(min_x, dataset->features().at(row, 0));
        max_x = std::max(max_x, dataset->features().at(row, 0));
        min_y = std::min(min_y, dataset->features().at(row, 1));
        max_y = std::max(max_y, dataset->features().at(row, 1));
    }

    const auto [x_min, x_max] = paddedRange(min_x, max_x);
    const auto [y_min, y_max] = paddedRange(min_y, max_y);
    const bool was_training = session_.model().isTraining();
    session_.model().eval();
    decision_grid_ = neuroforge::buildDecisionBoundaryGrid(session_.model(), *dataset, x_min, x_max, y_min, y_max, 30, 30);

    if (was_training) {
        session_.model().train();
    } else {
        session_.model().eval();
    }
}

void VisualApp::drawControls() {
    if (ImGui::Button("Reset XOR Demo")) {
        session_.resetXorDemo();
        refreshSnapshots();
    }

    ImGui::SameLine();
    ImGui::TextUnformatted(session_.status().c_str());

    ImGui::Separator();
    ImGui::InputText("Model path", model_path_.data(), model_path_.size());
    ImGui::SameLine();
    if (ImGui::Button("Load Model")) {
        session_.loadModel(model_path_.data());
        refreshSnapshots();
    }

    ImGui::InputText("CSV path", csv_path_.data(), csv_path_.size());
    ImGui::InputInt("Label column", &label_column_);
    ImGui::Checkbox("Has header", &has_header_);
    ImGui::InputInt("Class count", &class_count_);
    ImGui::SameLine();
    if (ImGui::Button("Load Dataset")) {
        session_.loadCsvDataset(csv_path_.data(), static_cast<size_t>(std::max(label_column_, 0)), has_header_);
        refreshSnapshots();
    }

    ImGui::Separator();
    ImGui::InputInt("Epochs", &epochs_);
    ImGui::InputFloat("Learning rate", &learning_rate_);
    ImGui::InputInt("Batch size", &batch_size_);
    ImGui::Checkbox("Shuffle", &shuffle_);
    ImGui::InputInt("Seed", &seed_);
    const char* losses[] = {"MSE", "BinaryCrossEntropy", "CrossEntropy"};
    const char* optimizers[] = {"SGD", "Adam"};
    ImGui::Combo("Loss", &loss_index_, losses, 3);
    ImGui::Combo("Optimizer", &optimizer_index_, optimizers, 2);

    if (ImGui::Button("Train")) {
        session_.trainCurrent(trainingConfig());
        refreshSnapshots();
    }
}

void VisualApp::drawPredictionView() const {
    ImGui::TextUnformatted("Predictions");

    if (!prediction_snapshot_.supported) {
        ImGui::TextUnformatted(prediction_snapshot_.message.c_str());
        return;
    }

    if (prediction_snapshot_.rows.empty()) {
        ImGui::TextUnformatted("No predictions");
        return;
    }

    const size_t limit = std::min<size_t>(prediction_snapshot_.rows.size(), 12);

    for (size_t row = 0; row < limit; ++row) {
        const neuroforge::PredictionRowSnapshot& item = prediction_snapshot_.rows[row];

        if (prediction_snapshot_.kind == neuroforge::PredictionKind::Multiclass) {
            ImGui::Text("row %zu predict class %zu target class %zu", row, item.predicted_class, item.target_class);
        } else if (prediction_snapshot_.kind == neuroforge::PredictionKind::Binary) {
            ImGui::Text("row %zu %.4f target %.4f", row, item.output[0], item.target[0]);
        } else {
            ImGui::Text("row %zu %.4f target %.4f", row, item.output[0], item.target.empty() ? 0.0 : item.target[0]);
        }
    }

    if (prediction_snapshot_.rows.size() > limit) {
        ImGui::Text("Showing %zu of %zu", limit, prediction_snapshot_.rows.size());
    }
}

neuroforge::VisualTrainingConfig VisualApp::trainingConfig() const {
    neuroforge::VisualTrainingConfig config;
    config.epochs = static_cast<size_t>(std::max(epochs_, 1));
    config.learning_rate = static_cast<double>(std::max(learning_rate_, 0.000001f));
    config.batch_size = static_cast<size_t>(std::max(batch_size_, 0));
    config.shuffle = shuffle_;
    config.seed = static_cast<uint32_t>(std::max(seed_, 0));
    config.class_count = static_cast<size_t>(std::max(class_count_, 1));
    config.loss = loss_index_ == 1
        ? neuroforge::VisualLossType::BinaryCrossEntropy
        : loss_index_ == 2
            ? neuroforge::VisualLossType::CrossEntropy
            : neuroforge::VisualLossType::MSE;
    config.optimizer = optimizer_index_ == 1 ? neuroforge::VisualOptimizerType::Adam : neuroforge::VisualOptimizerType::SGD;
    return config;
}

void VisualApp::drawFrame() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("NeuroForge Visual Lab", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    drawControls();

    ImGui::Separator();
    ImGui::Columns(3, "dashboard_columns", true);
    visual_lab::drawModelGraphView(network_graph_, model_snapshot_);
    ImGui::NextColumn();
    visual_lab::drawLossPlotView(loss_snapshot_);
    ImGui::NextColumn();
    drawPredictionView();
    ImGui::Columns(1);

    ImGui::Separator();
    ImGui::Columns(2, "dataset_columns", true);
    visual_lab::drawDatasetScatterView(dataset_snapshot_);
    ImGui::NextColumn();
    visual_lab::drawDecisionBoundaryView(decision_grid_);
    ImGui::Columns(1);

    ImGui::Separator();
    ImGui::Columns(2, "inspection_columns", true);
    visual_lab::drawTensorInspectorView(tensor_snapshots_);
    ImGui::NextColumn();
    visual_lab::drawGradientView(gradient_snapshots_);
    ImGui::Columns(1);
    ImGui::End();
}
