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

## Current Completion Target

NeuroForge is complete for the current educational roadmap when:

- default build passes
- all tests pass
- XOR example runs
- CSV regression example runs
- optional Visual Lab builds
- docs match actual behavior

## Future Work

Future work should happen only after the current project is stable:

- Adam optimizer
- binary cross entropy
- broader Tensor operations
- limited broadcasting
- gradient checking utilities
- DataLoader integration in `Trainer`
- model checkpoint format with optimizer state
- Graphviz export
- richer Visual Lab controls
- more examples

## Out Of Scope

- CUDA
- production ML performance
- external ML backends
- full PyTorch compatibility
- transformer framework
