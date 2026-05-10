#pragma once

#include "neuroforge/visualization/VisualizationData.hpp"

#include <vector>

namespace visual_lab {

void drawTensorInspectorView(const std::vector<neuroforge::TensorSnapshot>& tensors);

}
