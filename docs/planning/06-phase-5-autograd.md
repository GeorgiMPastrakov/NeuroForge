# Phase 5: Autograd Engine

## Goal

Add a clear educational autograd path after manual training already works.

Use `$neuroforge-autograd-engine`.

## Ticket 1

Task:
Implement scalar `Value` autograd.

Context:
Scalar autograd is easier to validate than tensor autograd and teaches graph traversal.

Goal:
Add `Value` with data, grad, parents, operations, and backward.

Files likely involved:
`include/neuroforge/autograd/Value.hpp`, `src/autograd/Value.cpp`, tests

Dependencies:
Phase 4.

Branch:
`autograd-engine`

Skill/agent:
`$neuroforge-autograd-engine`

Constraints:
Support addition, subtraction, multiplication, division, power, tanh, ReLU, sigmoid.

Expected output:
Scalar expressions produce correct gradients.

Tests required:
Known derivative cases, accumulation through multiple paths, zero_grad.

Definition of done:
Scalar autograd tests pass with exact expected gradients where possible.

Risks:
Shared ownership cycles can leak if graph ownership is sloppy.

## Ticket 2

Task:
Add tensor autograd metadata.

Context:
Tensor operations need optional gradient tracking.

Goal:
Extend Tensor with `requires_grad`, gradient storage, operation metadata, and backward hooks.

Files likely involved:
`Tensor.hpp`, `Tensor.cpp`, `include/neuroforge/autograd`, tests

Dependencies:
Ticket 1 and stable Math Core.

Branch:
`autograd-engine`

Skill/agent:
`$neuroforge-autograd-engine`

Constraints:
Do not break non-autograd Tensor use. Gradient shape must match data shape.

Expected output:
Tensor can track simple operation graphs.

Tests required:
Requires-grad flag, gradient storage, zero_grad, backward from scalar tensor.

Definition of done:
Existing tests still pass and new metadata tests pass.

Risks:
Retrofitting autograd into Tensor can destabilize Math Core.

## Ticket 3

Task:
Implement tensor operation gradients.

Context:
Autograd must support the operations used by small neural networks.

Goal:
Add gradients for add, subtract, element-wise multiply, matmul, sum, mean, ReLU, sigmoid, tanh, and MSE.

Files likely involved:
`Tensor.cpp`, `include/neuroforge/autograd`, tests

Dependencies:
Ticket 2.

Branch:
`autograd-engine`

Skill/agent:
`$neuroforge-autograd-engine`

Constraints:
Only support exact-shape operations already implemented. No broadcasting gradients.

Expected output:
Tensor backward computes expected gradients.

Tests required:
Known gradient values, matrix multiplication gradient, activation gradients, MSE gradient.

Definition of done:
All supported operations have gradient tests.

Risks:
Matmul gradient indexing is easy to get wrong.

## Ticket 4

Task:
Connect autograd to model training.

Context:
Manual training already works, but framework should demonstrate `loss.backward()`.

Goal:
Add an autograd-backed training path without deleting manual backward until parity is proven.

Files likely involved:
`Trainer`, losses, Tensor/autograd integration tests

Dependencies:
Ticket 3.

Branch:
`autograd-engine`

Skill/agent:
`$neuroforge-autograd-engine` and `$neuroforge-training-loop`

Constraints:
Do not pretend unsupported operations work. Keep manual path available if needed.

Expected output:
At least one tiny model trains or produces parameter gradients through autograd.

Tests required:
`loss.backward()` populates parameter gradients. Tiny training loss decreases.

Definition of done:
Autograd path is real, tested, and documented as supported for specific operations.

Risks:
Full PyTorch-like behavior is out of scope.
