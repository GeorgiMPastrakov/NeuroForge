# NeuroForge Architecture

NeuroForge is a CPU-only educational C++ machine learning framework. It is built to show the internal parts of a small PyTorch-like system without depending on external ML libraries.

## Module Map

```text
Core -> NN/Loss/Optim -> Training -> Examples
  |         |              |
  |         |              -> Visualization snapshots
  |         -> Serialization
  -> Data and Metrics
```

## Core

The Core module owns `Shape`, `Tensor`, and `Random`.

`Tensor` is the main numeric container. It supports rank-1 and rank-2 shapes, row-major storage, element-wise operations, matrix multiplication, reductions, activations, and limited autograd metadata.

## Neural Network

The NN module owns `Parameter`, `Module`, `Linear`, activations, and `Sequential`.

`Sequential` stores `std::shared_ptr<Module>` layers and exposes non-owning `Parameter*` values to optimizers. Manual backward remains part of the `Module` interface because NeuroForge supports a clear pre-autograd training path.

## Losses And Optimizers

Losses compute scalar training objectives and support manual backward. Supported losses are `MSELoss`, `MAELoss`, `BinaryCrossEntropyLoss`, and `CrossEntropyLoss`.

Optimizers update `Parameter` values from stored gradients. Supported optimizers are `SGD`, `MomentumSGD`, and `Adam`. Learning rate belongs to the optimizer.

## Training

`Trainer` connects a model, loss, optimizer, and optional `DataLoader`.

`fit` uses the manual backward path. `fitAutograd` uses Tensor autograd, then syncs Tensor gradients back into `Parameter` gradients before optimizer update.

`TrainingHistory` stores epoch and loss data for logs, tests, and visualization.

## Data, Metrics, Serialization

The data module provides strict numeric CSV loading, dataset slicing, batching, and train/test splitting.

The metrics module provides MSE, MAE, accuracy, and binary accuracy helpers.

Serialization supports saving and loading `Sequential` models containing `Linear`, `ReLU`, `Sigmoid`, and `Tanh`.

## Visualization

Visualization is optional. The core framework does not depend on Dear ImGui, ImPlot, GLFW, or OpenGL.

The non-GUI visualization data layer builds model, tensor, gradient, loss, and decision-boundary snapshots. The Visual Lab reads those snapshots and renders them in a desktop GUI.

## Current Limits

NeuroForge is intentionally small:

- CPU-only
- rank-1 and rank-2 tensors only
- no full broadcasting
- no CUDA
- no external ML backend
- no full PyTorch-compatible autograd
- no arbitrary model serialization
