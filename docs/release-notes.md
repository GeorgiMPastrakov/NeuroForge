# Release Notes

## NeuroForge Educational Release

This release completes the current NeuroForge roadmap.

## Implemented

- CPU-only Tensor math
- rank-1 and rank-2 shapes
- element-wise operations
- matrix multiplication
- reductions and activations
- absolute value, logarithm, clamp, LeakyReLU, and row softmax Tensor operations
- deterministic random initialization
- neural network modules including LeakyReLU, Softmax, and Dropout
- `Sequential` models
- manual backward training
- scalar and Tensor autograd
- MSE, MAE, BCE, and CrossEntropy losses
- SGD, MomentumSGD, and Adam optimizers
- DataLoader-aware Trainer overloads
- numeric CSV loading
- batches and train/test split
- MSE, MAE, accuracy, and binary accuracy metrics
- Sequential save/load
- XOR demo
- CSV regression demo
- dense classification demo
- optional generalized Visual Lab
- visual dense network graph
- planning and final documentation

## Validation

The release validation sequence is documented in `docs/validation.md`.

Minimum required checks:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
./build/examples/neuroforge_xor
./build/examples/neuroforge_linear_regression
./build/examples/neuroforge_dense_classification
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
```

## Known Limits

- CPU-only
- rank-1 and rank-2 Tensor support
- no full broadcasting
- no CUDA
- no external ML backend
- no arbitrary custom model serialization
- no optimizer checkpointing
- Visual Lab supports dense `Sequential` models only
- Visual Lab scatter and decision-boundary views require 2D features
- no CNNs
- no arbitrary custom module visualization

## Stretch Goals

- gradient checker
- broader Tensor operation coverage
- asynchronous Visual Lab training
- native file dialogs
- Graphviz export
- more demos
