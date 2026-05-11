# Phase 10: General Visual Lab

## Goal

Turn NeuroForge Visual Lab from a fixed XOR inspector into a general dashboard for supported NeuroForge models and datasets.

The network architecture view must become visual, not text-only. It should draw layers, neurons, and connections for dense `Sequential` models.

## Recommended Branch

`general-visual-lab`

## Dependency Rule

Phase 10 starts after Phase 9 planning is accepted. It can start before all Phase 9 features are implemented if it targets the current supported model set first:

- `Sequential`
- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`
- supported saved model format
- numeric CSV datasets

## Ticket 1

Task:
Add a Visual Lab session model.

Context:
Visual Lab currently builds one deterministic XOR demo state at startup.

Goal:
Introduce a session state that can hold the active model, dataset, training history, predictions, gradients, selected paths, and UI status.

Files likely involved:
`tools/visual_lab`, `include/neuroforge/visualization`, `src/visualization`

Dependencies:
Existing Visual Lab and snapshot builders.

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Keep GUI code separate from core training code. Do not make core modules depend on ImGui.

Expected output:
Visual Lab can switch from hardcoded XOR state to a reusable session state.

Tests required:
Snapshot unit tests for empty session, XOR session, loaded model session.

Definition of done:
The fixed XOR demo is represented as a session, not hardcoded directly in every view.

Risks:
Session state can become a dumping ground if ownership is not clear.

## Ticket 2

Task:
Load saved models.

Context:
Serialization supports `Sequential` models with `Linear`, `ReLU`, `Sigmoid`, and `Tanh`.

Goal:
Add GUI controls to load a saved model path and refresh architecture, tensor, and gradient views.

Files likely involved:
`tools/visual_lab`, `src/serialization`, `docs/visualization.md`

Dependencies:
`ModelLoader`

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`, `$neuroforge-nn-layers`

Constraints:
Support only the current saved `Sequential` format. Show clear UI error text on load failure.

Expected output:
Users can inspect a saved supported model in Visual Lab.

Tests required:
Load valid saved model, reject invalid path, reject unsupported file, snapshots update after load.

Definition of done:
Loaded models drive architecture and tensor views without hardcoded XOR assumptions.

Risks:
File dialogs are platform-sensitive. A text path input is acceptable first.

## Ticket 3

Task:
Load CSV datasets.

Context:
`CSVDataset` supports numeric CSV loading with selectable label column.

Goal:
Add GUI controls for dataset path, header option, label column, and load action.

Files likely involved:
`tools/visual_lab`, `src/data`, `docs/visualization.md`

Dependencies:
`CSVDataset`, `Dataset`

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`, `$neuroforge-training-loop`

Constraints:
Dataset scatter and decision boundary require exactly two feature columns. Higher-dimensional datasets should still load for tensor inspection and training logs, but scatter/boundary should show a clear unsupported message.

Expected output:
Visual Lab can inspect supported CSV datasets beyond XOR.

Tests required:
Valid CSV load, invalid label column, missing file, 2D scatter supported, high-dimensional scatter disabled.

Definition of done:
Dataset-dependent views read the loaded dataset, not a hardcoded XOR tensor.

Risks:
Bad CSV errors need to be readable in the GUI.

## Ticket 4

Task:
Add generic training controls.

Context:
Visual Lab should be able to train the currently loaded supported model and dataset.

Goal:
Add controls for epochs, learning rate, training path, and train/reset actions.

Files likely involved:
`tools/visual_lab`, `src/training`

Dependencies:
`Trainer`, `TrainingConfig`, `MSELoss`, `SGD`

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`, `$neuroforge-training-loop`

Constraints:
Start with MSE and SGD only. Manual and autograd training can be a toggle if both are reliable for the loaded model. GUI must not freeze for long runs; short synchronous training is acceptable first, but long training should be capped or moved to a worker later.

Expected output:
Users can train a loaded dense model on a loaded dataset and see updated loss, tensors, gradients, and predictions.

Tests required:
Training updates history, parameter snapshots change, invalid model/dataset shape gives clear error, reset restores demo state.

Definition of done:
Training controls work for XOR and a small CSV regression model.

Risks:
Generic training can expose shape mismatches that the fixed demo never hit.

## Ticket 5

Task:
Build a visual network graph.

Context:
The architecture view currently reads real layer metadata but is not a neuron graph.

Goal:
Render dense `Sequential` models as a visual graph with neurons and connections.

Files likely involved:
`tools/visual_lab/ModelGraphView.*`, `include/neuroforge/visualization/VisualizationData.hpp`, `src/visualization/VisualizationData.cpp`

Dependencies:
Layer metadata and `Linear` shape inspection.

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`, `$neuroforge-nn-layers`

Constraints:
Only draw neuron-level graphs for dense `Linear` networks. Use compact aggregation when layers are large. Activations should appear as layer bands or badges between Linear layers. Do not pretend unsupported layers have neuron-level structure.

Expected output:
Visual graph with:

- input nodes
- hidden/output neurons
- edges between adjacent dense layers
- layer labels
- parameter counts
- optional edge coloring or thickness based on weights for small networks

Tests required:
Graph layout snapshot data for empty model, XOR model, larger model requiring aggregation, unsupported layer fallback.

Definition of done:
The model architecture view is visibly a network graph, not only text.

Risks:
Rendering every edge for large layers can become unreadable. Cap edge drawing and summarize large layers.

## Ticket 6

Task:
Generalize tensor, gradient, loss, dataset, and boundary views.

Context:
Existing views are useful but demo-shaped.

Goal:
Make every view read only from session snapshots and handle missing or unsupported data.

Files likely involved:
`tools/visual_lab`, `src/visualization`

Dependencies:
Session model and snapshot builders.

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`

Constraints:
Do not crash on empty model, empty dataset, no history, no gradients, high-dimensional features, or non-binary outputs.

Expected output:
Visual Lab remains stable while switching models and datasets.

Tests required:
Empty state, model-only state, dataset-only state, trained state, high-dimensional dataset state.

Definition of done:
All views degrade cleanly when data is unavailable.

Risks:
GUI state bugs can hide because most unit tests cover only snapshot builders.

## Ticket 7

Task:
Update docs and validation.

Context:
General Visual Lab changes how users interact with the project.

Goal:
Update README, visualization docs, examples docs, and validation docs.

Files likely involved:
`README.md`, `docs/visualization.md`, `docs/examples.md`, `docs/validation.md`

Dependencies:
Tickets 1 through 6.

Branch:
`general-visual-lab`

Skill/agent:
`$neuroforge-visualization-lab`, documentation agent

Constraints:
Docs must state the exact model and dataset limits.

Expected output:
Users know how to load models, load CSVs, train, and interpret the graph view.

Tests required:
Documented commands build and run.

Definition of done:
Visual Lab docs match current GUI behavior.

Risks:
Overclaiming generality will make the tool look broken when unsupported models are loaded.

## Phase 10 Definition Of Done

- Visual Lab can load supported saved `Sequential` models.
- Visual Lab can load numeric CSV datasets.
- Visual Lab can train supported dense models from the GUI.
- Architecture view draws a real neuron/connection graph for dense networks.
- Large layers are summarized instead of rendered unreadably.
- Views handle unsupported states without crashing.
- Default core build still works without GUI dependencies.
- Visualizer build passes with `NEUROFORGE_BUILD_VISUALIZER=ON`.
