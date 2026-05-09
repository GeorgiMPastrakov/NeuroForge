# Phase 2: Math Core

## Goal

Build the tensor and shape foundation needed by all later modules.

Use `$neuroforge-math-core`.

## Ticket 1

Task:
Implement `Shape`.

Context:
Tensor operations need consistent shape validation and readable shape errors.

Goal:
Add a small `Shape` type with rank, rows, cols, size, equality, and string formatting.

Files likely involved:
`include/neuroforge/core/Shape.hpp`, `src/core/Shape.cpp`, `tests/test_shape.cpp`

Dependencies:
Phase 1.

Branch:
`math-core-tensor`

Skill/agent:
`$neuroforge-math-core`

Constraints:
Support 1D and 2D shapes first. Throw clear exceptions for invalid row/column access.

Expected output:
Reusable shape validation type.

Tests required:
Rank, size, rows, columns, equality, invalid dimension access, string format.

Definition of done:
All shape tests pass.

Risks:
Overengineering shape for N-dimensional views will delay Tensor work.

## Ticket 2

Task:
Implement Tensor storage and indexing.

Context:
Tensor is the main data object.

Goal:
Add `Tensor` with row-major `std::vector<double>` storage, shape, factories, indexing, and data access.

Files likely involved:
`include/neuroforge/core/Tensor.hpp`, `src/core/Tensor.cpp`, `tests/test_tensor.cpp`

Dependencies:
Ticket 1.

Branch:
`math-core-tensor`

Skill/agent:
`$neuroforge-math-core`

Constraints:
Support 1D and 2D tensors first. No broadcasting. No views or strides.

Expected output:
`Tensor::fromVector`, zeros, ones, full, shape access, element access.

Tests required:
Creation, shape storage, size, 1D indexing, 2D indexing, invalid indexing.

Definition of done:
Tensor construction and indexing are tested and clear.

Risks:
Changing storage later will affect all modules, so keep the first API small.

## Ticket 3

Task:
Implement basic tensor operations.

Context:
NN layers and losses need element-wise math.

Goal:
Add addition, subtraction, element-wise multiplication, scalar multiplication, sum, mean, and transpose.

Files likely involved:
`Tensor.hpp`, `Tensor.cpp`, `tests/test_tensor.cpp`

Dependencies:
Ticket 2.

Branch:
`math-core-tensor`

Skill/agent:
`$neuroforge-math-core`

Constraints:
Operands must have identical shapes except scalar operations. Throw `std::invalid_argument` on mismatches.

Expected output:
Element-wise operations with clear shape errors.

Tests required:
Valid operations, invalid shape operations, transpose values, sum, mean.

Definition of done:
Every operation has normal and invalid-case tests.

Risks:
Silent broadcasting would hide bugs in training.

## Ticket 4

Task:
Implement matrix multiplication.

Context:
`Linear` depends on `Tensor::matmul`.

Goal:
Add correct O(n^3) 2D matrix multiplication.

Files likely involved:
`Tensor.hpp`, `Tensor.cpp`, `tests/test_tensor.cpp`

Dependencies:
Ticket 2.

Branch:
`math-core-tensor`

Skill/agent:
`$neuroforge-math-core`

Constraints:
Only 2D matmul. Shape rule is `[m, n] * [n, p] = [m, p]`.

Expected output:
Known matrix products compute correctly.

Tests required:
2x2 by 2x2, 2x3 by 3x1, invalid shape throws with clear message.

Definition of done:
Matmul tests pass and error messages include both shapes.

Risks:
Wrong flat indexing will poison every later layer.

## Ticket 5

Task:
Implement activations and deterministic random initialization.

Context:
XOR needs nonlinear activations and repeatable initialization.

Goal:
Add ReLU, sigmoid, tanh, and a simple deterministic random helper.

Files likely involved:
`include/neuroforge/core/Random.hpp`, `src/core/Random.cpp`, `Tensor.hpp`, `Tensor.cpp`, tests

Dependencies:
Tickets 2 and 3.

Branch:
`math-core-tensor`

Skill/agent:
`$neuroforge-math-core`

Constraints:
Use standard C++ random tools. No external math libraries.

Expected output:
Deterministic tensor initialization and activation operations.

Tests required:
Seed reproducibility, ReLU values, sigmoid values, tanh values.

Definition of done:
Activation and random tests pass deterministically.

Risks:
Non-deterministic tests will make training failures hard to debug.
