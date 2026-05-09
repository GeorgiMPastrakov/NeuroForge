# Phase 3: Neural Network, Loss, Optimizer

## Goal

Build the minimal model API and update path needed for manual training.

Use `$neuroforge-nn-layers` for layers. Use `$neuroforge-training-loop` only for integration decisions touching losses and optimizers.

## Ticket 1

Task:
Implement `Parameter` and `Module`.

Context:
Trainable layers need a shared interface for parameters and gradient clearing.

Goal:
Add `Parameter` with data, gradient, name, and `zero_grad`. Add abstract `Module`.

Files likely involved:
`include/neuroforge/nn/Parameter.hpp`, `include/neuroforge/nn/Module.hpp`, `src/nn`, tests

Dependencies:
Phase 2.

Branch:
`nn-loss-optim`

Skill/agent:
`$neuroforge-nn-layers`

Constraints:
No raw owning pointers. `parameters()` returns non-owning `Parameter*`.

Expected output:
Base trainable parameter model.

Tests required:
Parameter stores data and grad, `zero_grad` clears grad, default module parameter list works.

Definition of done:
Parameter ownership and gradient clearing are tested.

Risks:
Bad ownership here will leak into optimizer and trainer.

## Ticket 2

Task:
Implement `Linear`.

Context:
Linear is the core neural network layer.

Goal:
Add `Linear(in_features, out_features)` with forward and manual backward support.

Files likely involved:
`include/neuroforge/nn/Linear.hpp`, `src/nn/Linear.cpp`, tests

Dependencies:
Ticket 1.

Branch:
`nn-loss-optim`

Skill/agent:
`$neuroforge-nn-layers`

Constraints:
Forward shape is `[batch, in] -> [batch, out]`. Bias addition must not depend on generic broadcasting.

Expected output:
Linear forward output and parameter gradients through manual backward.

Tests required:
Forward shape, known-value forward, invalid input shape, parameter count, backward gradient shapes.

Definition of done:
Linear can be used by `Sequential` and trainer.

Risks:
Incorrect bias gradient across batches will break training.

## Ticket 3

Task:
Implement activation modules.

Context:
XOR requires nonlinear layers.

Goal:
Add `ReLU`, `Sigmoid`, and `Tanh` modules with forward and manual backward support.

Files likely involved:
`include/neuroforge/nn/ReLU.hpp`, `Sigmoid.hpp`, `Tanh.hpp`, `src/nn`, tests

Dependencies:
Phase 2 and Ticket 1.

Branch:
`nn-loss-optim`

Skill/agent:
`$neuroforge-nn-layers`

Constraints:
No trainable parameters. Output shape equals input shape.

Expected output:
Activation modules usable in a model.

Tests required:
Forward values, backward values, parameter list is empty.

Definition of done:
Activations pass numeric tests and support backward chaining.

Risks:
Sigmoid saturation can slow XOR training if initialization is poor.

## Ticket 4

Task:
Implement `Sequential`.

Context:
Users should compose models with ordered layers.

Goal:
Add `Sequential::add`, `forward`, `backward`, `parameters`, and `summary`.

Files likely involved:
`include/neuroforge/nn/Sequential.hpp`, `src/nn/Sequential.cpp`, tests

Dependencies:
Tickets 1 through 3.

Branch:
`nn-loss-optim`

Skill/agent:
`$neuroforge-nn-layers`

Constraints:
Store `std::shared_ptr<Module>`. Backward order must be reverse forward order.

Expected output:
Composable model object with recursive parameter collection.

Tests required:
Forward shape, parameter collection, summary contains layer names, backward order works.

Definition of done:
XOR model can be constructed.

Risks:
Parameter collection bugs will make optimizer update nothing.

## Ticket 5

Task:
Implement `MSELoss`.

Context:
XOR and regression need a simple loss.

Goal:
Add MSE forward returning scalar loss and backward returning prediction gradient.

Files likely involved:
`include/neuroforge/losses/LossFunction.hpp`, `MSELoss.hpp`, `src/losses`, tests

Dependencies:
Phase 2.

Branch:
`nn-loss-optim`

Skill/agent:
Training integration agent.

Constraints:
Prediction and target shapes must match exactly.

Expected output:
MSE loss works without autograd.

Tests required:
Known MSE value, gradient values, shape mismatch throws.

Definition of done:
MSE can drive manual backward training.

Risks:
Wrong loss derivative prevents learning.

## Ticket 6

Task:
Implement `SGD`.

Context:
Parameters need updates from gradients.

Goal:
Add optimizer base and SGD update rule.

Files likely involved:
`include/neuroforge/optim/Optimizer.hpp`, `SGD.hpp`, `src/optim`, tests

Dependencies:
Ticket 1.

Branch:
`nn-loss-optim`

Skill/agent:
Training integration agent.

Constraints:
Throw if optimizer receives no parameters.

Expected output:
`SGD(parameters, learning_rate)` updates parameter data and clears gradients when requested.

Tests required:
Known update, zero_grad, empty parameter list throws.

Definition of done:
SGD can update `Linear` parameters.

Risks:
Gradient shape mismatch must not be ignored.
