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
- generalized Visual Lab session loading and training
- visual dense network graph
- Visual Lab support for binary, regression, and multiclass snapshots

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

## Phase 10: General Visual Lab

Completed:

- load supported saved `Sequential` models
- load numeric CSV datasets
- train supported dense models from the GUI
- generalize all views around a session state
- render dense network architecture as a real neuron graph with connections
- summarize large layers instead of drawing unreadable edge clouds
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
