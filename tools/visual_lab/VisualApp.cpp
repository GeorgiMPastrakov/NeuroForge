#include "VisualApp.hpp"

#include "GradientView.hpp"
#include "LossPlotView.hpp"
#include "ModelGraphView.hpp"
#include "TensorInspectorView.hpp"
#include "imgui.h"
#include "implot.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>

#include <string>

VisualApp::VisualApp()
    : dataset_(neuroforge::Tensor::fromVector({{0.0, 0.0}}), neuroforge::Tensor::fromVector({{0.0}})),
      prediction_(neuroforge::Tensor::fromVector({0.0})) {
    buildDemoState();
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

void VisualApp::buildDemoState() {
    neuroforge::Random::seed(42);
    model_ = neuroforge::Sequential();
    model_.add(std::make_shared<neuroforge::Linear>(2, 4));
    model_.add(std::make_shared<neuroforge::Sigmoid>());
    model_.add(std::make_shared<neuroforge::Linear>(4, 1));
    model_.add(std::make_shared<neuroforge::Sigmoid>());

    dataset_ = neuroforge::Dataset(
        neuroforge::Tensor::fromVector({
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
        }),
        neuroforge::Tensor::fromVector({
            {0.0},
            {1.0},
            {1.0},
            {0.0}
        })
    );

    neuroforge::MSELoss loss;
    neuroforge::SGD optimizer(model_.parameters(), 1.0);
    neuroforge::Trainer trainer(model_, loss, optimizer);
    neuroforge::TrainingConfig config;
    config.epochs = 3000;
    history_ = trainer.fit(dataset_.features(), dataset_.labels(), config);
    prediction_ = model_.forward(dataset_.features());
    refreshSnapshots();
}

void VisualApp::refreshSnapshots() {
    model_snapshot_ = neuroforge::buildModelSnapshot(model_);
    loss_snapshot_ = neuroforge::buildLossHistorySnapshot(history_);
    gradient_snapshots_ = neuroforge::buildGradientSnapshots(model_, 12);
    tensor_snapshots_.clear();

    for (neuroforge::Parameter* parameter : model_.parameters()) {
        tensor_snapshots_.push_back(neuroforge::buildTensorSnapshot(parameter->name(), parameter->data(), 12));
    }
}

void VisualApp::drawFrame() {
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("NeuroForge Visual Lab", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse);

    if (ImGui::Button("Reset XOR Demo")) {
        buildDemoState();
    }

    ImGui::Separator();
    ImGui::Columns(3, "dashboard_columns", true);

    visual_lab::drawModelGraphView(model_snapshot_);

    ImGui::NextColumn();
    visual_lab::drawLossPlotView(loss_snapshot_);

    ImGui::NextColumn();
    ImGui::TextUnformatted("Predictions");
    for (size_t row = 0; row < dataset_.size(); ++row) {
        ImGui::Text("[%.0f, %.0f] -> %.4f target %.0f",
            dataset_.features().at(row, 0),
            dataset_.features().at(row, 1),
            prediction_.at(row, 0),
            dataset_.labels().at(row, 0));
    }

    ImGui::Columns(1);
    ImGui::Separator();
    ImGui::Columns(2, "inspection_columns", true);
    visual_lab::drawTensorInspectorView(tensor_snapshots_);
    ImGui::NextColumn();
    visual_lab::drawGradientView(gradient_snapshots_);
    ImGui::Columns(1);
    ImGui::End();
}
