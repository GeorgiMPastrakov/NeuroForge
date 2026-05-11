# NeuroForge Visual Lab

NeuroForge Visual Lab is an optional desktop GUI for inspecting the framework.

The core library builds without GUI dependencies.

## Build

```bash
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
./build-visual/tools/visual_lab/neuroforge_visual_lab
```

## Dependencies

The visualizer uses:

- Dear ImGui
- ImPlot
- GLFW
- OpenGL

These dependencies are fetched only when `NEUROFORGE_BUILD_VISUALIZER=ON`.

## Views

Current views:

- model architecture
- training loss plot
- tensor inspector
- gradient monitor
- dataset scatter plot
- decision boundary view

## Data Flow

The GUI reads real data from:

- `Sequential`
- `TrainingHistory`
- `Tensor`
- `Dataset`
- model predictions

Non-GUI snapshot builders live in the framework so they can be tested without opening a window.

## Limits

The current Visual Lab builds a deterministic XOR demo state at startup. It is a demonstration and inspection tool, not a full experiment manager.
