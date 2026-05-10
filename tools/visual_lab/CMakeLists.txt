include(FetchContent)

find_package(OpenGL REQUIRED)

set(NEUROFORGE_GIT_BASE "https:/")

FetchContent_Declare(
    glfw
    GIT_REPOSITORY ${NEUROFORGE_GIT_BASE}/github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY ${NEUROFORGE_GIT_BASE}/github.com/ocornut/imgui.git
    GIT_TAG v1.92.7
)

FetchContent_Declare(
    implot
    GIT_REPOSITORY ${NEUROFORGE_GIT_BASE}/github.com/epezent/implot.git
    GIT_TAG v1.0
)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(glfw imgui implot)

add_library(neuroforge_imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    ${implot_SOURCE_DIR}/implot.cpp
    ${implot_SOURCE_DIR}/implot_items.cpp
)

target_include_directories(neuroforge_imgui
    PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${implot_SOURCE_DIR}
)

target_link_libraries(neuroforge_imgui PUBLIC glfw OpenGL::GL)

add_executable(neuroforge_visual_lab
    main.cpp
    VisualApp.cpp
    ModelGraphView.cpp
    LossPlotView.cpp
    GradientView.cpp
    TensorInspectorView.cpp
    DatasetScatterView.cpp
    DecisionBoundaryView.cpp
)

target_link_libraries(neuroforge_visual_lab PRIVATE neuroforge neuroforge_imgui)

target_compile_definitions(neuroforge_visual_lab PRIVATE GL_SILENCE_DEPRECATION)
