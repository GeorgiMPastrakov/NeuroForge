---
name: neuroforge-autograd-engine
description: Use this skill when designing, implementing, debugging, or reviewing NeuroForge's autograd system, computational graph, backward pass, gradient accumulation, operation nodes, and gradient correctness tests.
---

# NeuroForge Autograd Engine Skill

## Purpose

You are the Autograd specialist for NeuroForge.

Your job is to design and implement automatic differentiation for a small C++ machine learning framework inspired by PyTorch.

Autograd is one of the most important parts of NeuroForge. It should show how `loss.backward()` works internally.

The goal is not to recreate full PyTorch autograd. The goal is to build a correct, understandable, educational autograd system that supports the operations needed for small neural networks.

---

## When To Use This Skill

Use this skill when the task involves:

- autograd
- `requires_grad`
- gradients
- `backward()`
- computation graph
- operation nodes
- graph traversal
- topological sorting
- gradient accumulation
- scalar `Value` autograd
- tensor autograd
- derivative rules
- gradient checking
- debugging wrong gradients
- connecting loss backward to parameters

Do not use this skill for:

- pure matrix operations without gradients
- GUI visualization
- optimizer update formulas, except when checking gradient flow
- CSV loading
- documentation-only tasks

---

## Main Goal

NeuroForge should eventually support code like:

```cpp
Tensor prediction = model.forward(input);
Tensor loss = loss_fn.forward(prediction, target);

loss.backward();

optimizer.step();
```

After `loss.backward()`, trainable parameters should contain gradients.

---

## Recommended Strategy

Do not try to build full tensor autograd immediately.

Use this order:

```text
1. Make Math Core correct.
2. Make basic neural network training work with manual backward if needed.
3. Implement a small scalar Value autograd engine.
4. Test scalar autograd heavily.
5. Extend the same ideas to Tensor operations.
6. Connect Tensor autograd to Linear layers and losses.
```

Correct training is more important than pretending to support full PyTorch-style autograd.

---

## Core Concepts

Autograd needs to track:

```text
value/data
gradient
requires_grad
parents
operation that created the value/tensor
local backward function
topological order
gradient accumulation
```

A tensor created from an operation should know enough to send gradients back to its inputs.

---

## Possible Classes

The Autograd module may include:

```text
Value
ComputationGraph
Operation
AddOperation
SubOperation
MulOperation
DivOperation
PowOperation
MatMulOperation
ReLUOperation
SigmoidOperation
TanhOperation
MSEOperation
BackwardEngine
GradientChecker
```

Keep the design understandable.

Do not create too many abstract classes before there is a working implementation.

---

## Scalar Value Autograd

A scalar `Value` class is useful for learning and testing.

Example:

```cpp
Value a(2.0, true);
Value b(3.0, true);

Value c = a * b;
Value d = c + a;

d.backward();
```

Expected gradients:

```text
d = a*b + a
dd/da = b + 1 = 4
dd/db = a = 2
```

This is easier to validate than tensor autograd.

---

## Tensor Autograd

Tensor autograd should support only the operations needed by NeuroForge MVP first:

```text
add
subtract
element-wise multiply
matmul
mean
sum
pow
relu
sigmoid
mse
```

Do not support every Tensor operation until the basic set is correct.

---

## Required Gradient Rules

### Addition

```text
z = x + y
dL/dx += dL/dz
dL/dy += dL/dz
```

### Subtraction

```text
z = x - y
dL/dx += dL/dz
dL/dy -= dL/dz
```

### Element-wise Multiplication

```text
z = x * y
dL/dx += dL/dz * y
dL/dy += dL/dz * x
```

### Matrix Multiplication

```text
Z = XW

dL/dX = dL/dZ * W^T
dL/dW = X^T * dL/dZ
```

### Mean

If:

```text
z = mean(x)
```

Then:

```text
dL/dx_i += dL/dz / number_of_elements
```

### Sum

If:

```text
z = sum(x)
```

Then:

```text
dL/dx_i += dL/dz
```

### ReLU

```text
relu(x) = max(0, x)

gradient:
1 if x > 0
0 if x <= 0
```

### Sigmoid

```text
sigmoid(x) = 1 / (1 + exp(-x))

gradient:
sigmoid(x) * (1 - sigmoid(x))
```

### Tanh

```text
tanh gradient:
1 - tanh(x)^2
```

### MSE

```text
MSE = mean((prediction - target)^2)

dMSE/dprediction = 2 * (prediction - target) / n
```

---

## Backward Pass Rules

When `backward()` is called:

```text
1. Set the output gradient to 1 if the output is scalar.
2. Build topological order from output to leaves.
3. Traverse nodes in reverse topological order.
4. Call each node's backward function.
5. Accumulate gradients into parent tensors.
```

Do not overwrite gradients incorrectly.

Gradients should accumulate unless `zero_grad()` has been called.

---

## Gradient Accumulation

If a tensor contributes to the loss through multiple paths, its gradient must accumulate.

Example:

```text
z = x*x + x
```

The gradient of `x` receives contributions from both `x*x` and `x`.

Use `+=`, not assignment, when adding gradient contributions.

---

## Shape Safety

Tensor gradient shapes must match tensor data shapes.

If a tensor has shape:

```text
[2, 4]
```

Its gradient should also have shape:

```text
[2, 4]
```

Throw clear errors if gradient shapes are incompatible.

---

## Required Tests

Add or update tests for:

```text
Value addition gradient
Value multiplication gradient
Value expression gradient
Tensor addition gradient
Tensor element-wise multiplication gradient
Matrix multiplication gradient
ReLU gradient
Sigmoid gradient
MSE gradient
Gradient accumulation through multiple paths
zero_grad clears gradients
loss.backward populates model parameter gradients
XOR training loss decreases with autograd if connected
```

Use known derivatives.

Use numerical gradient checking for critical operations when possible.

---

## Numerical Gradient Checking

For a function `f(x)`, approximate derivative:

```text
df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

Use small epsilon:

```text
eps = 1e-5
```

Compare with autograd gradient using tolerance:

```text
1e-4 or 1e-5 depending on numerical stability
```

---

## Definition Of Done

An Autograd task is done when:

```text
The graph is built correctly.
backward() computes expected gradients.
Gradients accumulate correctly.
zero_grad works.
Tests cover known derivative cases.
Shape errors are handled clearly.
The implementation is understandable.
The implementation does not fake gradients.
```

---

## Review Checklist

Before finishing, check:

```text
Are gradients accumulated, not overwritten?
Does backward order follow the graph correctly?
Are repeated nodes handled safely?
Does loss.backward() work from a scalar loss?
Do parameter gradients match parameter shapes?
Are there tests for the exact derivative?
Does this support Linear, activation, and MSE training?
```

---

## Important Warning

Do not hide broken autograd behind manual updates.

If autograd is incomplete, clearly document which operations support it.

It is acceptable to have:

```text
manual backprop training path
+
experimental autograd path
```

during development.

It is not acceptable to pretend full autograd works when it does not.
