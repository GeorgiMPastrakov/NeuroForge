# Tensor And Shape

`Shape` and `Tensor` are the math foundation of NeuroForge.

## Shape

`Shape` supports rank 1 and rank 2.

Supported behavior:

- rank
- total size
- row and column access for rank-2 shapes
- dimension access
- equality checks
- readable string output

Invalid shapes throw exceptions:

- empty shape
- zero dimension
- rank greater than 2
- row or column access on rank-1 shape

## Tensor Storage

`Tensor` stores `std::vector<double>` data in row-major order.

Supported construction:

- raw data plus `Shape`
- 1D vector
- 2D vector
- zeros
- ones
- full

2D construction rejects ragged input.

## Operations

Supported operations:

- indexing with bounds checks
- exact-shape add, subtract, and element-wise multiply
- scalar multiply
- matrix multiplication for rank-2 tensors
- row-vector addition for layer bias handling
- transpose for rank-2 tensors
- sum and mean
- absolute value
- logarithm
- clamp
- power
- ReLU
- LeakyReLU
- sigmoid
- tanh
- row-wise softmax for rank-2 tensors
- scalar extraction with `item`

No general broadcasting exists. Element-wise operations require exact shape matches.

## Autograd Metadata

`Tensor` can track:

- `requiresGrad`
- gradient storage
- parents from supported operations
- backward functions for supported operations

`Tensor::backward()` starts only from scalar tensors.

## Design Limits

NeuroForge does not implement:

- arbitrary-rank tensors
- views
- strides
- slicing
- full broadcasting
- BLAS kernels
- SIMD kernels
