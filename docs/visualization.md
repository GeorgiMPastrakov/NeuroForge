# NeuroForge Visual Lab

NeuroForge Visual Lab is an optional desktop GUI for explaining one trained XOR classifier.

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

- visual XOR network graph
- training loss curve
- combined XOR decision map and truth-table samples
- compact prediction table

The architecture view draws the `2 -> 4 -> 1` network as neurons and weighted connections. Green edges are positive weights, red edges are negative weights, and thicker edges have larger magnitudes.

## Data Flow

The GUI reads real data through `VisualSession`:

- deterministic XOR `Sequential`
- XOR `Dataset`
- `TrainingHistory`
- model predictions
- visualization snapshots

Non-GUI snapshot builders live in the framework so they can be tested without opening a window.

## Demo

Visual Lab automatically builds and trains:

```text
Linear(2 -> 4) -> ReLU -> Linear(4 -> 1) -> Sigmoid
```

The dashboard is display-only. It does not expose model loading, CSV loading, or training controls.

## Limits

The GUI intentionally presents one XOR classifier for a clear class demonstration. The framework still contains reusable snapshot builders and session APIs, but the desktop app does not expose generic experiment management.
