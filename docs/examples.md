# Examples

NeuroForge includes two command-line examples and one optional GUI demo.

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

## Visual Lab

Build and run:

```bash
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
./build-visual/tools/visual_lab/neuroforge_visual_lab
```

The visualizer shows a deterministic XOR model, training loss, tensors, gradients, dataset points, and decision boundary data.
