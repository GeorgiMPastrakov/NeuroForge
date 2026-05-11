# Autograd

NeuroForge has two autograd surfaces:

- scalar `Value`
- limited Tensor autograd

The goal is educational clarity, not PyTorch completeness.

## Scalar Value

`Value` tracks scalar data, gradient, parents, and a backward function.

Supported scalar operations:

- addition
- subtraction
- multiplication
- division
- power
- ReLU
- sigmoid
- tanh

Calling `backward()` builds a topological order and accumulates gradients into leaf values that require gradients.

Gradients accumulate until `zeroGrad()` is called.

## Tensor Autograd

Tensor autograd supports the operations needed by the current framework:

- add
- subtract
- element-wise multiply
- scalar multiply
- matrix multiplication
- row-vector bias addition
- sum
- mean
- absolute value
- logarithm
- clamp
- power
- ReLU
- LeakyReLU
- row-wise softmax
- sigmoid
- tanh
- MSE loss
- MAE loss
- binary cross entropy loss
- cross entropy loss

`Tensor::backward()` requires a scalar tensor.

## Model Integration

`Parameter::enableAutograd()` marks trainable Tensor data as gradient-tracked.

`Parameter::syncAutogradGrad()` copies Tensor autograd gradients into the `Parameter` gradient Tensor used by optimizers.

`Trainer::fitAutograd()` runs:

```text
zero gradients
enable autograd
forward
loss
loss.backward
sync parameter gradients
optimizer step
record loss
```

## Manual Backward Still Exists

Manual backward remains available through `Trainer::fit`.

This is intentional. It keeps the educational training path clear and provides a stable path separate from the newer Tensor autograd path.

## Limits

Tensor autograd does not support:

- full broadcasting
- all Tensor operations
- arbitrary custom operations
- non-scalar backward roots
- advanced graph memory controls
