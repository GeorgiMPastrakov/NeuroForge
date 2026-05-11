# NeuroForge Visual Lab

NeuroForge Visual Lab is an optional desktop GUI for inspecting and training supported NeuroForge models.

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

- visual dense network graph
- training loss plot
- tensor inspector
- gradient monitor
- dataset scatter plot
- decision boundary view
- prediction table

The architecture view draws dense `Sequential` networks as neurons and weighted connections. Large layers are summarized so the view stays readable.

## Data Flow

The GUI reads real data through `VisualSession`:

- active `Sequential`
- optional numeric `Dataset`
- `TrainingHistory`
- model predictions
- tensor and gradient snapshots

Non-GUI snapshot builders live in the framework so they can be tested without opening a window.

## Loading

Visual Lab supports text-path loading for:

- saved `Sequential` models
- numeric CSV datasets

Supported loaded layers are:

- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `LeakyReLU`
- `Softmax`
- `Dropout`

CSV loading uses the existing `CSVDataset` rules. Values must be numeric. The label column is selected in the GUI.

## Training

GUI training is synchronous and uses the manual `Trainer::fit` path.

Supported GUI losses:

- MSE
- BinaryCrossEntropy
- CrossEntropy

Supported GUI optimizers:

- SGD
- Adam

For CrossEntropy, scalar CSV labels are converted from integer class indices to one-hot rows using the GUI class-count setting.

## Limits

Supported model type is `Sequential` dense models only.

Scatter and decision-boundary views require 2D features. Non-2D datasets load, but those views show an unsupported-state message.

Visual Lab does not support CNNs, N-D tensors, arbitrary custom modules, native file dialogs, asynchronous training, or optimizer checkpoint loading.
