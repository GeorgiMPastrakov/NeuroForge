# NeuroForge

NeuroForge is a small educational C++ machine learning framework inspired by PyTorch.

The goal is to build tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, serialization, and visualization tools from scratch.

## Current Status

Phase 1 is in progress.

Implemented:

- CMake project skeleton
- public umbrella header
- smoke test
- planned module folder layout

Not implemented yet:

- Tensor
- Shape
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

## Roadmap

The next phase is Math Core:

- `Shape`
- `Tensor`
- indexing
- element-wise operations
- transpose
- matrix multiplication
- deterministic random initialization
- core math tests

Planning docs live in `docs/planning/`.

## Constraints

NeuroForge is CPU-only and educational. It is not a PyTorch replacement, not a CUDA framework, and not a wrapper around external machine learning libraries.
