# Release Notes

## NeuroForge Educational Release

This release completes the current NeuroForge roadmap.

## Implemented

- CPU-only Tensor math
- rank-1 and rank-2 shapes
- element-wise operations
- matrix multiplication
- reductions and activations
- deterministic random initialization
- neural network modules
- `Sequential` models
- manual backward training
- scalar and Tensor autograd
- MSE loss
- SGD optimizer
- numeric CSV loading
- batches and train/test split
- MSE, MAE, accuracy, and binary accuracy metrics
- Sequential save/load
- XOR demo
- CSV regression demo
- optional Visual Lab
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
- Visual Lab is a demo inspector, not a full experiment manager

## Stretch Goals

- Adam
- binary cross entropy
- gradient checker
- broader Tensor operation coverage
- DataLoader-aware Trainer loops
- richer Visual Lab controls
- Graphviz export
- more demos
