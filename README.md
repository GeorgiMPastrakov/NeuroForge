# NeuroForge

NeuroForge is a small educational C++ machine learning framework inspired by PyTorch. It implements tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, data loading, serialization, and visualization from scratch.

The goal is to show how a CPU-only ML framework works internally.

## Features

| Area | Implemented |
| --- | --- |
| Core math | `Shape`, `Tensor`, indexing, factories, element-wise ops, transpose, matmul, reductions, activations |
| Autograd | scalar `Value`, Tensor autograd for supported ops, `loss.backward()` path |
| Neural networks | `Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Sequential` |
| Loss and optimization | `MSELoss`, `SGD` |
| Training | manual training, autograd training, loss history, evaluation loss |
| Data | numeric CSV loading, datasets, batches, train/test split |
| Metrics | MSE, MAE, accuracy, binary accuracy |
| Serialization | save/load for supported `Sequential` models |
| Examples | XOR classifier, CSV regression |
| Visualization | optional NeuroForge Visual Lab with model, loss, tensor, gradient, dataset, and decision boundary views |

## Build And Test

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

## Public Include

```cpp
#include "neuroforge/neuroforge.hpp"
```

## Minimal Example

```cpp
#include "neuroforge/neuroforge.hpp"

#include <memory>

using namespace neuroforge;

int main() {
    Random::seed(42);

    Tensor X = Tensor::fromVector({
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}
    });

    Tensor y = Tensor::fromVector({
        {0.0},
        {1.0},
        {1.0},
        {0.0}
    });

    Sequential model;
    model.add(std::make_shared<Linear>(2, 4));
    model.add(std::make_shared<Sigmoid>());
    model.add(std::make_shared<Linear>(4, 1));
    model.add(std::make_shared<Sigmoid>());

    MSELoss loss;
    SGD optimizer(model.parameters(), 0.5);

    TrainingConfig config;
    config.epochs = 5000;

    Trainer trainer(model, loss, optimizer);
    TrainingHistory history = trainer.fit(X, y, config);

    Tensor prediction = model.forward(X);

    return history.finalLoss() < 0.3 && prediction.shape() == Shape({4, 1}) ? 0 : 1;
}
```

## Run Examples

Build first:

```bash
cmake -S . -B build
cmake --build build
```

Run XOR:

```bash
./build/examples/neuroforge_xor
```

Run CSV regression:

```bash
./build/examples/neuroforge_linear_regression
```

The regression demo saves a trained model to:

```text
build/neuroforge_linear_regression_model.txt
```

## Visual Lab

Build and run the optional desktop visualizer:

```bash
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
./build-visual/tools/visual_lab/neuroforge_visual_lab
```

The default framework build does not require GUI dependencies. Visual Lab fetches Dear ImGui, ImPlot, and GLFW only when enabled.

## Documentation

- [Architecture](docs/architecture.md)
- [Tensor And Shape](docs/tensor.md)
- [Autograd](docs/autograd.md)
- [Training](docs/training.md)
- [Data And Metrics](docs/data.md)
- [Serialization](docs/serialization.md)
- [Visualization](docs/visualization.md)
- [Examples](docs/examples.md)
- [Validation](docs/validation.md)
- [Roadmap](docs/roadmap.md)
- [Release Notes](docs/release-notes.md)
- [Planning Docs](docs/planning/00-overview.md)

## Constraints

NeuroForge is:

- CPU-only
- educational
- built from scratch
- focused on small models and clear internals

NeuroForge is not:

- a PyTorch replacement
- a CUDA framework
- a production ML system
- a transformer framework
- a wrapper around external ML libraries

## Current Limits

- Tensors are focused on rank 1 and rank 2.
- Broadcasting is intentionally limited.
- Autograd supports only the implemented operation set.
- Serialization supports the current `Sequential` layer set only.
- Visual Lab is an inspection demo, not a full experiment manager.
