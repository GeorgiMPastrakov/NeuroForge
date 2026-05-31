# Roadmap

## Completed

- CMake project skeleton
- Core `Shape`, `Tensor`, and `Random`
- tensor math and activations
- `Parameter`, `Module`, `Linear`, activations, and `Sequential`
- `MSELoss`
- `SGD`
- manual training loop
- XOR example
- scalar `Value` autograd
- Tensor autograd for supported operations
- autograd training path
- numeric CSV loading
- dataset batching and splitting
- metrics
- Sequential save/load
- CSV regression example
- optional NeuroForge Visual Lab
- final project documentation
- MAE, BCE, and CrossEntropy losses
- MomentumSGD and Adam optimizers
- LeakyReLU, Softmax, and Dropout
- DataLoader-aware Trainer overloads
- dense classification example
- reusable Visual Lab session loading and training backend
- visual dense network graph
- reusable binary, regression, and multiclass visualization snapshots
- focused display-only XOR Visual Lab dashboard

## Current Completion Target

NeuroForge is complete for the current educational roadmap when:

- default build passes
- all tests pass
- XOR example runs
- CSV regression example runs
- optional Visual Lab builds
- docs match actual behavior

## Future Work

Future work should happen only after the current project is stable.

## Phase 10: Visual Lab Backend And XOR Dashboard

Completed:

- keep reusable session APIs for supported saved models, CSV datasets, and dense-model training
- keep generalized non-GUI snapshot builders
- present one deterministic XOR classifier in the desktop app
- render its dense network architecture as a real neuron graph with weighted connections
- show loss convergence, decision regions, truth-table samples, and predictions
- keep the core framework independent from GUI dependencies

## Later Major Work

- N-D `Shape` and `Tensor`
- reshape and flatten
- convolution layers
- pooling layers
- convolution autograd
- image datasets
- CNN examples
- gradient checking utilities
- model checkpoint format with optimizer state
- Graphviz export
- more examples

## Out Of Scope

- CUDA
- production ML performance
- external ML backends
- full PyTorch compatibility
- transformer framework
