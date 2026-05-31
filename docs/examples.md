# Examples

NeuroForge includes three command-line examples and one optional GUI demo.

## XOR

Build:

```bash
cmake -S . -B build
cmake --build build
```

Run:

```bash
./build/examples/neuroforge_xor
```

The XOR demo uses:

- `Sequential`
- `Linear`
- `Sigmoid`
- `MSELoss`
- `SGD`
- `Trainer`

It prints initial loss, final loss, model summary, and predictions.

## CSV Regression

Run:

```bash
./build/examples/neuroforge_linear_regression
```

The regression demo uses:

- `CSVDataset`
- `TrainTestSplitter`
- `Linear`
- `MSELoss`
- `SGD`
- metrics
- model save/load

It saves a trained model to:

```text
build/neuroforge_linear_regression_model.txt
```

## Dense Classification

Run:

```bash
./build/examples/neuroforge_dense_classification
```

The dense classification demo uses:

- hardcoded 2D three-class data
- one-hot targets
- `Linear`
- `LeakyReLU`
- `CrossEntropyLoss`
- `Adam`
- `DataLoader`

It prints initial loss, final loss, accuracy, and class predictions.

## Visual Lab

Build and run:

```bash
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
./build-visual/tools/visual_lab/neuroforge_visual_lab
```

The display-only visualizer automatically trains and presents a deterministic XOR model. It shows a neuron graph, MSE convergence curve, combined decision map with truth-table points, and compact prediction table.
