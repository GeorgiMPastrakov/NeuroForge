---
name: neuroforge-math-core
description: Use this skill when implementing or reviewing NeuroForge's Tensor, Matrix, Shape, random initialization, indexing, transpose, element-wise operations, matrix multiplication, and numerical math foundations in C++.
---

# NeuroForge Math Core Skill

## Purpose

You are the Math Core specialist for NeuroForge.

Your job is to build and protect the mathematical foundation of the framework.

NeuroForge is a small C++ machine learning framework inspired by PyTorch. The Math Core module provides the basic structures and operations needed by every other module: tensors, matrices, shapes, indexing, matrix multiplication, transposition, initialization, and numerical utility functions.

Do not treat this as a generic math utility library. This module exists specifically to support machine learning and neural network training.

---

## When To Use This Skill

Use this skill when the task involves:

- `Tensor`
- `Matrix`
- `Shape`
- tensor/matrix storage
- tensor indexing
- shape validation
- matrix multiplication
- transpose
- element-wise operations
- scalar operations
- random initialization
- zeros/ones/full tensors
- math correctness tests
- numerical stability
- tensor printing/debugging
- shape mismatch errors

Do not use this skill for:

- neural network layer design, unless the issue is purely tensor math
- optimizer design
- training loops
- GUI visualization
- CSV parsing
- model serialization

---

## Main Goal

The Math Core should make it possible to write simple code like:

```cpp
Tensor a = Tensor::fromVector({
    {1.0, 2.0},
    {3.0, 4.0}
});

Tensor b = Tensor::fromVector({
    {5.0, 6.0},
    {7.0, 8.0}
});

Tensor c = a.matmul(b);
```

The system must validate shapes, compute correct results, and throw clear errors when operations are invalid.

---

## Design Principles

Follow these principles:

1. Correctness before performance.
2. Clear shape validation before computation.
3. Simple APIs before advanced abstractions.
4. 1D and 2D tensors first.
5. No full PyTorch broadcasting at the beginning.
6. No external math libraries for core ML operations.
7. Use `std::vector<double>` for storage.
8. Use row-major layout unless the project explicitly chooses otherwise.
9. Throw clear `std::invalid_argument` errors for invalid shapes.
10. Write tests for every operation.

---

## Suggested Classes

The Math Core may include:

```text
Tensor
Matrix
Shape
Random
Initializer
TensorPrinter
```

Prefer `Tensor` as the long-term main object.

`Matrix` may exist as an early helper, but do not duplicate too much logic between `Matrix` and `Tensor`.

---

## Tensor Responsibilities

`Tensor` should eventually support:

```text
data storage
shape storage
rank
size
indexing
reshape
transpose for 2D tensors
element-wise addition
element-wise subtraction
element-wise multiplication
scalar operations
matrix multiplication
sum
mean
pow
exp
log
relu
sigmoid
tanh
gradient storage if needed by autograd
requires_grad flag if needed by autograd
```

Start with the features required by XOR and regression.

---

## Suggested Tensor Fields

A simple first design:

```cpp
class Tensor {
private:
    std::vector<double> data_;
    std::vector<size_t> shape_;
    std::vector<double> grad_;
    bool requires_grad_ = false;

public:
    Tensor();
};
```

Keep ownership simple.

Avoid raw owning pointers.

---

## Shape Rules

Always check shapes before operations.

Examples:

### Addition

Valid:

```text
[2, 3] + [2, 3]
```

Invalid:

```text
[2, 3] + [3, 2]
```

Do not silently broadcast unless broadcasting is explicitly implemented and tested.

### Matrix Multiplication

Valid:

```text
[A, B] matmul [B, C] = [A, C]
```

Invalid:

```text
[A, B] matmul [D, C] where B != D
```

Good error message:

```text
MatMul shape mismatch: left shape [4, 2] cannot multiply right shape [3, 1].
```

Bad error message:

```text
Invalid operation.
```

---

## Matrix Multiplication Requirements

Implement simple correct O(n^3) matrix multiplication first.

For:

```text
C = A * B
A shape: [m, n]
B shape: [n, p]
C shape: [m, p]
```

Formula:

```text
C[i][j] = sum over k of A[i][k] * B[k][j]
```

Test with known values.

Example:

```text
A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

A * B = [[19, 22],
         [43, 50]]
```

---

## Required Tests

Add or update tests for:

```text
Tensor creation
Shape storage
Total size calculation
Indexing
Invalid indexing
Addition
Subtraction
Element-wise multiplication
Scalar multiplication
Transpose
Matrix multiplication 2x2 * 2x2
Matrix multiplication 2x3 * 3x1
Invalid matmul shape throws
Zeros initialization
Ones initialization
Random initialization with deterministic seed
Sigmoid numerical output
ReLU numerical output
Mean and sum if implemented
```

Use deterministic random seeds.

---

## Definition Of Done

A Math Core task is done when:

```text
The operation is implemented.
Shape validation exists.
Clear errors are thrown for invalid shapes.
Tests cover normal and invalid cases.
The API is clean.
The implementation does not depend on external ML/math libraries.
The code builds cleanly.
```

---

## Review Checklist

Before finishing, check:

```text
Does this operation preserve correct shape?
Does it throw on invalid shape?
Are indexes converted correctly into flat vector positions?
Are tests deterministic?
Is the implementation readable?
Is the function name clear?
Will this support Linear layers and training later?
```

---

## Important Warning

Do not overbuild the Tensor system too early.

Avoid these until the MVP works:

```text
full broadcasting
N-dimensional optimized kernels
SIMD
CUDA
BLAS
complex slicing
views
strides
automatic mixed precision
```

Start simple. Make it correct. Make it testable.
