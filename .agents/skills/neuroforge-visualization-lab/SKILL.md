---
name: neuroforge-visualization-lab
description: Use this skill when designing, implementing, or reviewing NeuroForge Visual Lab, including Dear ImGui, ImPlot, model architecture visualization, training loss plots, tensor inspection, gradient monitoring, dataset scatter plots, and decision boundary views.
---

# NeuroForge Visualization Lab Skill

## Purpose

You are the Visualization specialist for NeuroForge.

Your job is to make NeuroForge visually impressive and easier to understand.

NeuroForge Visual Lab should be a graphical C++ dashboard that visualizes what happens inside the framework: model architecture, training loss, gradients, tensor values, datasets, and predictions.

The visualizer must stay separate from the core ML framework.

The core library must work without the GUI.

---

## When To Use This Skill

Use this skill when the task involves:

- NeuroForge Visual Lab
- Dear ImGui
- ImPlot
- visual dashboard
- model architecture view
- training loss graph
- gradient monitor
- tensor inspector
- dataset scatter plot
- decision boundary visualization
- real-time training visualization
- GUI separation from core ML

Do not use this skill for:

- core tensor math
- autograd internals
- optimizer formulas
- non-visual training logic
- CSV parsing, except displaying loaded datasets

---

## Main Goal

Create a real graphical tool, not a terminal-only output.

The visualizer should eventually show:

```text
Model architecture
Training loss curve
Gradient norms
Tensor values
Dataset scatter plot
Decision boundary for 2D classifiers
```

Preferred technologies:

```text
Dear ImGui
ImPlot
CMake
```

---

## Architecture Rule

The visualizer must be optional.

The dependency direction should be:

```text
Visualization depends on Core/NN/Training
Core/NN/Training do not depend on Visualization
```

Good:

```text
Trainer writes TrainingHistory.
VisualApp reads TrainingHistory.
```

Bad:

```text
Trainer directly calls ImGui functions.
```

Never mix GUI rendering logic with training logic.

---

## Suggested Classes

The visualization module may include:

```text
VisualApp
ModelGraphView
LossPlotView
GradientView
TensorInspectorView
DatasetScatterView
DecisionBoundaryView
TrainingDashboard
```

Start with:

```text
VisualApp
ModelGraphView
LossPlotView
```

Then add more views.

---

## Visual Feature 1: Model Architecture View

Show the model as connected blocks.

Example:

```text
Input → Linear(2→4) → ReLU → Linear(4→1) → Sigmoid → Output
```

Each block should show:

```text
layer name
input shape
output shape
parameter count
```

This view should use model metadata, not hardcoded strings.

The neural network module should expose enough summary data for this view.

---

## Visual Feature 2: Training Loss Plot

Show loss over epochs.

This should read from `TrainingHistory`.

Example data:

```text
epoch 1, loss 0.69
epoch 50, loss 0.42
epoch 100, loss 0.21
epoch 200, loss 0.05
```

Use ImPlot if available.

The plot should update during or after training.

Minimum acceptable version:

```text
Train first, then show completed loss curve.
```

Better version:

```text
Show live updates while training runs.
```

Do not block the GUI forever during long training if live updates are implemented.

---

## Visual Feature 3: Gradient Monitor

Show gradient norms per layer.

Example:

```text
Linear1 weights grad norm: 0.042
Linear1 bias grad norm: 0.008
Linear2 weights grad norm: 0.017
Linear2 bias grad norm: 0.004
```

This helps prove backpropagation works.

The training module should expose gradient data through `TrainingHistory` or model inspection methods.

---

## Visual Feature 4: Tensor Inspector

Allow inspecting tensor values.

Example display:

```text
Tensor: Linear1.weights
Shape: [2 x 4]

0.21  -0.43   0.12   0.88
0.10   0.56  -0.31   0.44
```

Useful tensors to inspect:

```text
layer weights
layer biases
layer gradients
input batch
prediction tensor
loss tensor
```

Keep large tensors scrollable or truncated.

---

## Visual Feature 5: Dataset Scatter Plot

For 2D datasets, display points.

Useful demos:

```text
XOR
2D point classifier
simple binary classification
```

Display:

```text
x coordinate
y coordinate
label/class
prediction if available
```

This makes the model behavior easier to explain.

---

## Visual Feature 6: Decision Boundary View

This is advanced but very impressive.

For 2D classifiers:

```text
1. Generate a grid of points.
2. Run model prediction on each point.
3. Color each region based on predicted class.
4. Draw real dataset points on top.
```

This proves the model learned a separation.

Only implement this after:

```text
model.forward works
2D classifier works
dataset scatter plot works
```

---

## Visual App Layout

Suggested layout:

```text
Top bar:
Run demo | Train | Reset | Save | Load

Left panel:
Model architecture

Center panel:
Dataset scatter / decision boundary

Right panel:
Tensor inspector / gradient monitor

Bottom panel:
Training loss plot and logs
```

Keep it clean.

Do not make the GUI too complex before the core visual features work.

---

## Data Interfaces Needed

The visualizer should consume clean data structures.

Useful inputs:

```text
TrainingHistory
ModelSummary
LayerInfo
Tensor
Dataset
Prediction grid
Gradient statistics
```

If these interfaces do not exist, suggest adding them to the non-GUI modules.

Do not directly inspect private fields with hacks.

---

## Build Rules

Visualization should be optional in CMake.

Suggested option:

```cmake
option(NEUROFORGE_BUILD_VISUALIZER "Build NeuroForge Visual Lab" ON)
```

If GUI dependencies are missing, the core framework and tests should still build.

---

## Required Tests / Checks

GUI code is harder to unit test, but still check:

```text
Core project builds without visualization.
Visualizer target builds when enabled.
TrainingHistory can provide plot data.
Model summary can provide layer data.
Tensor inspector handles small tensors.
Dataset scatter receives valid points.
```

Manual visual checks:

```text
Window opens.
Model architecture appears.
Loss plot shows data.
No crash when training history is empty.
No crash when model has no layers.
```

---

## Definition Of Done

A visualization task is done when:

```text
The visual feature works in a real window.
The core framework still builds without GUI.
The GUI reads clean data from framework interfaces.
Rendering logic is separate from training logic.
The visual output helps explain the model.
The code is not hardcoded only for one demo unless explicitly marked as a demo.
```

---

## Review Checklist

Before finishing, check:

```text
Does this add unnecessary dependency to core code?
Can the core tests run without GUI?
Is the displayed data real?
Is the layout understandable?
Does this help explain NeuroForge?
Can the visualizer be extended later?
Is the visualizer reading TrainingHistory or model metadata cleanly?
```

---

## Important Warning

Do not let the visualizer take over the project.

NeuroForge is primarily a machine learning framework.

The visualizer is a powerful demonstration and debugging tool, but it must not make the core architecture messy.

The best design is:

```text
Clean ML framework
+
Optional visual dashboard
```
