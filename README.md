# NeuroForge

NeuroForge is a small educational C++ machine learning framework inspired by PyTorch.

The goal is to build tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, serialization, and visualization tools from scratch.

## Current Status

Phase 6 Data, Metrics, and Serialization is implemented.

Implemented:

- CMake project skeleton
- public umbrella header
- smoke test
- planned module folder layout
- `Shape`
- `Tensor`
- tensor indexing
- tensor factories
- element-wise tensor operations
- transpose
- matrix multiplication
- sum and mean
- ReLU, sigmoid, and tanh
- deterministic random initialization
- core math tests
- `Parameter`
- `Module`
- `Linear`
- `ReLU`, `Sigmoid`, and `Tanh` modules
- `Sequential`
- `MSELoss`
- `SGD`
- `TrainingConfig`
- `TrainingHistory`
- `Trainer`
- XOR training example
- deterministic training behavior tests
- scalar `Value` autograd
- Tensor gradient metadata
- Tensor operation gradients
- MSE autograd path
- parameter gradient sync
- `Trainer::fitAutograd`
- `Dataset`
- `CSVDataset`
- `DataLoader`
- `TrainTestSplitter`
- MSE, MAE, accuracy, and binary accuracy metrics
- Sequential model save/load
- CSV regression example

Not implemented yet:

- visualization

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Run the XOR demo:

```bash
./build/examples/neuroforge_xor
```

Run the CSV regression demo:

```bash
./build/examples/neuroforge_linear_regression
```

## Public Include

```cpp
#include "neuroforge/neuroforge.hpp"
```

## Example

```cpp
#include "neuroforge/neuroforge.hpp"

#include <memory>

using namespace neuroforge;

int main() {
    Sequential model;

    model.add(std::make_shared<Linear>(2, 4));
    model.add(std::make_shared<ReLU>());
    model.add(std::make_shared<Linear>(4, 1));
    model.add(std::make_shared<Sigmoid>());

    Tensor input = Tensor::fromVector({
        {0.0, 1.0}
    });

    Tensor output = model.forward(input);

    return output.shape() == Shape({1, 1}) ? 0 : 1;
}
```

## Roadmap

The next phase is NeuroForge Visual Lab:

- optional GUI dashboard
- model architecture view
- loss curve
- tensor and gradient inspection
- dataset scatter plots

## Autograd Support

Supported scalar `Value` operations:

- addition
- subtraction
- multiplication
- division
- power
- ReLU
- sigmoid
- tanh

Supported Tensor autograd operations:

- add
- subtract
- element-wise multiply
- scalar multiply
- matrix multiplication
- row-vector bias addition
- sum
- mean
- power
- ReLU
- sigmoid
- tanh
- MSE loss

Manual backward training remains available through `Trainer::fit`. The autograd path is available through `Trainer::fitAutograd`.

Planning docs live in `docs/planning/`.

## Constraints

NeuroForge is CPU-only and educational. It is not a PyTorch replacement, not a CUDA framework, and not a wrapper around external machine learning libraries.
