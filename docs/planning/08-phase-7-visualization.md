# Phase 7: NeuroForge Visual Lab

## Goal

Add an optional graphical dashboard that visualizes real framework data without coupling GUI logic to core ML code.

Use `$neuroforge-visualization-lab`.

## Ticket 1

Task:
Add optional visualizer build target.

Context:
The core library must build without GUI dependencies.

Goal:
Add a CMake option for Visual Lab and a minimal app target.

Files likely involved:
`CMakeLists.txt`, `src/visualization`, `include/neuroforge/visualization`

Dependencies:
Phase 4.

Branch:
`visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Core tests must pass with visualizer disabled.

Expected output:
Optional visualizer target exists.

Tests required:
Core build with visualizer disabled. Visualizer build when dependencies are available.

Definition of done:
GUI dependencies do not affect core build.

Risks:
GUI setup can consume time before data interfaces are ready.

## Ticket 2

Task:
Expose model and training metadata.

Context:
Visualization needs clean data from non-GUI modules.

Goal:
Ensure model summary, layer info, parameter counts, and training history are available through public interfaces.

Files likely involved:
`Sequential`, `Module`, `TrainingHistory`, visualization interfaces

Dependencies:
Phase 4.

Branch:
`visual-lab`

Skill/agent:
`$neuroforge-visualization-lab` with NN/training review.

Constraints:
Core modules must not include GUI headers.

Expected output:
Visualizer can read architecture and loss data.

Tests required:
Model summary data test, history empty and populated cases.

Definition of done:
No private-field hacks are needed by the GUI.

Risks:
If metadata is added directly for the GUI, core APIs can become messy.

## Ticket 3

Task:
Implement model architecture and loss plot views.

Context:
These are the visual MVP features.

Goal:
Show model blocks and training loss curve from real data.

Files likely involved:
`ModelGraphView`, `LossPlotView`, `VisualApp`

Dependencies:
Tickets 1 and 2.

Branch:
`visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Display real `Sequential` and `TrainingHistory` data.

Expected output:
Window opens and shows architecture plus loss plot.

Tests required:
Manual visual check, no crash on empty model, no crash on empty history.

Definition of done:
Visual MVP explains model structure and training progress.

Risks:
Hardcoded demo text would reduce value.

## Ticket 4

Task:
Implement gradient and tensor inspection views.

Context:
Gradients and tensors prove internal training behavior.

Goal:
Show parameter values, gradients, shapes, and gradient norms.

Files likely involved:
`GradientView`, `TensorInspectorView`

Dependencies:
Ticket 2.

Branch:
`visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Large tensors should be truncated or scrollable.

Expected output:
Users can inspect weights, biases, and gradients.

Tests required:
Manual visual check with trained XOR model.

Definition of done:
Tensor and gradient data shown are real.

Risks:
Exposing too much mutable internal state can damage architecture.

## Ticket 5

Task:
Implement dataset scatter and decision boundary views.

Context:
2D classification visuals make the framework impressive and understandable.

Goal:
Show XOR or 2D classifier points and learned decision boundary.

Files likely involved:
`DatasetScatterView`, `DecisionBoundaryView`

Dependencies:
Phase 6 and Tickets 1 through 4.

Branch:
`visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Decision boundary samples grid points and runs `model.forward`.

Expected output:
Visualizer displays dataset points and learned regions.

Tests required:
Manual visual check with XOR or 2D dataset.

Definition of done:
Decision regions reflect model predictions, not hardcoded labels.

Risks:
This should not block the earlier visual MVP.
