# Phase 4: Training Loop and XOR MVP

## Goal

Make NeuroForge train a real small model and prove loss decreases.

Use `$neuroforge-training-loop`.

## Ticket 1

Task:
Implement `TrainingConfig` and `TrainingHistory`.

Context:
Training needs configuration and recorded loss values for logs and later visualization.

Goal:
Add config defaults and history storage for epoch loss and optional metrics.

Files likely involved:
`include/neuroforge/training/TrainingConfig.hpp`, `TrainingHistory.hpp`, `src/training`, tests

Dependencies:
Phase 3.

Branch:
`training-xor`

Skill/agent:
`$neuroforge-training-loop`

Constraints:
Keep history simple. Minimum required field is loss per epoch.

Expected output:
History can store, expose, and report final loss.

Tests required:
Config defaults, add loss, final loss, empty history behavior.

Definition of done:
History is usable by trainer and future Visual Lab.

Risks:
If history is too thin, visualization will need refactors later.

## Ticket 2

Task:
Implement `Trainer`.

Context:
Users need a high-level training loop.

Goal:
Add `Trainer` that runs forward, loss, zero_grad, backward, optimizer step, and history logging.

Files likely involved:
`include/neuroforge/training/Trainer.hpp`, `src/training/Trainer.cpp`, tests

Dependencies:
Phase 3 and Ticket 1.

Branch:
`training-xor`

Skill/agent:
`$neuroforge-training-loop`

Constraints:
Trainer must not hardcode XOR. It should work with tensors, model, loss, and optimizer.

Expected output:
`trainer.fit(X, y, config)` returns `TrainingHistory`.

Tests required:
One epoch runs, parameters change, history records loss, invalid shape propagates clear error.

Definition of done:
Trainer works with `Sequential`, `MSELoss`, and `SGD`.

Risks:
Wrong zero_grad order can hide gradient bugs.

## Ticket 3

Task:
Add XOR example.

Context:
XOR proves hidden layers and nonlinear activation matter.

Goal:
Add an executable example that trains a 2-4-1 network on XOR and prints predictions.

Files likely involved:
`examples/xor.cpp`, `CMakeLists.txt`

Dependencies:
Tickets 1 and 2.

Branch:
`training-xor`

Skill/agent:
`$neuroforge-training-loop`

Constraints:
Predictions must come from the trained model. Use deterministic seed.

Expected output:
Example shows loss decreasing and final predictions for four XOR inputs.

Tests required:
Example builds. Manual run produces sensible output.

Definition of done:
XOR example trains without hardcoded predictions.

Risks:
SGD may need learning-rate or initialization tuning.

## Ticket 4

Task:
Add XOR training test.

Context:
The project needs stable proof that training works.

Goal:
Add deterministic test proving final XOR loss is lower than initial loss.

Files likely involved:
`tests/test_training.cpp`

Dependencies:
Tickets 1 through 3.

Branch:
`training-xor`

Skill/agent:
Testing agent with `$neuroforge-training-loop`.

Constraints:
Do not require perfect XOR accuracy if it makes tests flaky.

Expected output:
Training test proves real loss decrease.

Tests required:
Initial loss greater than final loss by a stable margin.

Definition of done:
`ctest` passes repeatedly.

Risks:
Too strict convergence criteria will create flaky tests.
