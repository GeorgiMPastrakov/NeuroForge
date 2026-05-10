#include "VisualApp.hpp"

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

    ImGui::TextUnformatted("Model");
    ImGui::Text("Layers: %zu", model_snapshot_.layers.size());
    ImGui::Text("Parameters: %zu", model_snapshot_.total_parameters);
    for (const neuroforge::LayerInfo& layer : model_snapshot_.layers) {
        ImGui::BulletText("%s (%zu)", layer.name.c_str(), layer.parameter_count);
    }

    ImGui::NextColumn();
    ImGui::TextUnformatted("Training");
    ImGui::Text("Epochs: %zu", loss_snapshot_.epochs.size());
    if (!loss_snapshot_.losses.empty()) {
        ImGui::Text("Final loss: %.6f", loss_snapshot_.losses.back());
    }
    if (!loss_snapshot_.losses.empty() && ImPlot::BeginPlot("Loss", ImVec2(-1, 260))) {
        std::vector<double> epochs;
        epochs.reserve(loss_snapshot_.epochs.size());
        for (size_t epoch : loss_snapshot_.epochs) {
            epochs.push_back(static_cast<double>(epoch));
        }
        ImPlot::PlotLine("MSE", epochs.data(), loss_snapshot_.losses.data(), static_cast<int>(loss_snapshot_.losses.size()));
        ImPlot::EndPlot();
    }

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
    ImGui::End();
}
