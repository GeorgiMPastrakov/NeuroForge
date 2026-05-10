# NeuroForge

NeuroForge is a small educational C++ machine learning framework inspired by PyTorch.

The goal is to build tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, serialization, and visualization tools from scratch.

## Current Status

Phase 3 Neural Network, Loss, and Optimizer is implemented.

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

Not implemented yet:

- training loop
- autograd
- data loading
- serialization
- visualization

## Build

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
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

The next phase is Training and XOR:

- `TrainingConfig`
- `TrainingHistory`
- `Trainer`
- XOR example
- XOR loss decrease test

Planning docs live in `docs/planning/`.

## Constraints

NeuroForge is CPU-only and educational. It is not a PyTorch replacement, not a CUDA framework, and not a wrapper around external machine learning libraries.
