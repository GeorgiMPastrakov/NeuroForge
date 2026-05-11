# NeuroForge Roadmap Overview

## Current State

The repository has completed Phases 1 through 8 and is ready for optional post-release feature phases.

Implemented so far:

- CMake skeleton and public umbrella header
- `Shape`, `Tensor`, math operations, and deterministic random initialization
- neural network layers, MSE loss, SGD, and manual backward
- trainer, training history, deterministic XOR demo, and training tests
- scalar `Value` autograd
- Tensor autograd metadata and operation gradients
- autograd-backed MSE, parameter sync, and `Trainer::fitAutograd`
- dataset and CSV loading
- deterministic batching and train/test splitting
- evaluation metrics
- Sequential model save/load
- CSV regression demo
- optional NeuroForge Visual Lab
- model architecture, loss, tensor, gradient, dataset, and decision boundary views
- final documentation, validation docs, and release notes

## Target

NeuroForge should become a small CPU-only C++ machine learning framework inspired by PyTorch. It must demonstrate tensors, matrix operations, neural network layers, losses, optimizers, training loops, serialization, and optional visualization from scratch.

## Phase Order

| Phase | Branch | Main Output | Blocks |
| --- | --- | --- | --- |
| 1 | `scaffold-project` | CMake skeleton, folders, README, smoke test | all later work |
| 2 | `math-core-tensor` | `Shape`, `Tensor`, math ops, tests | NN, losses, training |
| 3 | `nn-loss-optim` | layers, losses, optimizer, manual backward | training |
| 4 | `training-xor` | trainer, history, XOR demo and test | MVP |
| 5 | `autograd-engine` | scalar and tensor autograd | advanced training path |
| 6 | `data-metrics-serialization` | CSV, dataloader, metrics, save/load | regression demo |
| 7 | `visual-lab` | optional GUI dashboard | visual MVP |
| 8 | `polish-release` | docs, examples, cleanup, final checks | release |
| 9 | `model-features` | more losses, optimizers, dense model features | richer dense ML |
| 10 | `general-visual-lab` | generic model/dataset Visual Lab and neuron graph | broader demos |

## Dependency Graph

```text
Phase 1 Skeleton
  -> Phase 2 Math Core
    -> Phase 3 NN + Loss + Optimizer
      -> Phase 4 Training + XOR MVP
        -> Phase 5 Autograd
        -> Phase 6 Data + Metrics + Serialization
          -> Phase 7 Visualization
            -> Phase 8 Polish + Release
              -> Phase 9 Expanded Model Features
                -> Phase 10 General Visual Lab
```

## MVP Definition

The MVP is complete when NeuroForge can:

- configure, build, and test with CMake
- create tensors
- validate shapes
- perform matrix multiplication
- define a `Sequential` model
- use `Linear`, `ReLU`, `Sigmoid`, and `Tanh`
- calculate MSE loss
- update parameters with SGD
- train XOR with real loss decrease
- store training history
- run all tests successfully

## Final Definition

The finished project should also include:

- scalar and tensor autograd
- CSV loading and batching
- regression demo
- metrics
- model save/load
- Visual Lab with architecture, loss, gradient, tensor, dataset, and decision boundary views
- clear README and architecture docs
- final reproducible validation commands

## Post-Release Direction

Phase 9 expands dense-network capabilities:

- MAE and BCE losses
- MomentumSGD and Adam
- LeakyReLU, Softmax, and CrossEntropyLoss
- DataLoader-aware Trainer overloads
- train/eval mode for Dropout and BatchNorm1d if included

Phase 10 generalizes Visual Lab:

- load supported saved `Sequential` models
- load numeric CSV datasets
- train supported dense models from the GUI
- render dense networks as visual neuron graphs with connections
- keep text summaries as supporting detail only

CNN support is not part of Phase 9 or Phase 10. It should be treated as a later major architecture phase because it requires N-D Tensor and convolution backward support.

## Non-Negotiable Rules

- No branch names using the forbidden Codex prefix from `AGENTS.md`.
- No comments in C++ code.
- No external ML libraries for framework logic.
- No fake predictions or hardcoded training results.
- Every core operation gets tests.
- Every phase must build and test before commit.

## Standard Checks

Every implementation phase must run:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Planning-only phases validate file existence and content instead.

## Agent Skill Routing

| Work | Skill |
| --- | --- |
| Tensor, Shape, math ops | `$neuroforge-math-core` |
| Autograd, graph, gradients | `$neuroforge-autograd-engine` |
| Layers, parameters, model API | `$neuroforge-nn-layers` |
| Trainer, history, XOR | `$neuroforge-training-loop` |
| GUI dashboard | `$neuroforge-visualization-lab` |

## Parallel Work Rule

Do not parallelize before Phase 1 is merged.

After Phase 1, Phase 2 should still be mostly single-threaded because Tensor APIs define the rest of the project.

After Phase 2, Phase 3 can split carefully:

- one agent for NN layers
- one agent for loss and optimizer
- one agent for tests if APIs are stable

Visualization, serialization, CSV, and autograd are not MVP blockers.
