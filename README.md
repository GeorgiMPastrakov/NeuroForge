# NeuroForge

NeuroForge is a small educational C++ machine learning framework inspired by PyTorch.

The goal is to build tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, serialization, and visualization tools from scratch.

## Current Status

Phase 2 Math Core is implemented.

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

Not implemented yet:

- neural network layers
- loss functions
- optimizers
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

using namespace neuroforge;

int main() {
    Tensor a = Tensor::fromVector({
        {1.0, 2.0},
        {3.0, 4.0}
    });

    Tensor b = Tensor::fromVector({
        {5.0, 6.0},
        {7.0, 8.0}
    });

    Tensor c = a.matmul(b);

    return c.at(0, 0) == 19.0 ? 0 : 1;
}
```

## Roadmap

The next phase is Neural Network, Loss, and Optimizer:

- `Parameter`
- `Module`
- `Linear`
- `ReLU`, `Sigmoid`, and `Tanh` modules
- `Sequential`
- `MSELoss`
- `SGD`

Planning docs live in `docs/planning/`.

## Constraints

NeuroForge is CPU-only and educational. It is not a PyTorch replacement, not a CUDA framework, and not a wrapper around external machine learning libraries.
