# Phase 9: Expanded Model Features

## Goal

Add the next practical dense-network features without breaking the clean educational architecture.

This phase improves NeuroForge as a small CPU-only ML framework. It does not start CNN support yet. CNNs require a larger Tensor redesign and should be planned as a later phase after Phase 9 and Phase 10 are stable.

## Recommended Branch

`model-features`

## Dependency Rule

Phase 9 starts after Phase 8 is merged and validated.

Keep this phase single-branch at first. Split into worktrees only after the first optimizer/loss milestone is green, because loss, optimizer, Trainer, docs, and examples share APIs.

## Ticket 1

Task:
Add more losses.

Context:
NeuroForge currently supports `MSELoss`. Dense binary and regression demos need a few more standard losses.

Goal:
Implement `MAELoss` and `BinaryCrossEntropyLoss`.

Files likely involved:
`include/neuroforge/losses`, `src/losses`, `tests`, `include/neuroforge/neuroforge.hpp`

Dependencies:
Existing `Tensor`, `LossFunction`, and manual/autograd training paths.

Branch:
`model-features`

Skill/agent:
`$neuroforge-training-loop`, `$neuroforge-autograd-engine`

Constraints:
Keep manual backward working. Add Tensor autograd support only when the needed Tensor operations already exist or are added with tests. Clamp BCE inputs internally for numerical stability.

Expected output:
`MAELoss` and `BinaryCrossEntropyLoss` usable in examples and tests.

Tests required:
Known forward values, known gradients, shape mismatch throws, BCE handles predictions close to 0 and 1 without NaN.

Definition of done:
Loss tests pass, existing training tests pass, docs list supported losses accurately.

Risks:
BCE can produce infinities if prediction values are not clamped.

## Ticket 2

Task:
Add optimizers.

Context:
Only SGD exists. Momentum and Adam make demos train faster and show more optimizer concepts.

Goal:
Implement `MomentumSGD` and `Adam`.

Files likely involved:
`include/neuroforge/optim`, `src/optim`, `tests`, `include/neuroforge/neuroforge.hpp`

Dependencies:
Existing `Parameter` and `Optimizer` interfaces.

Branch:
`model-features`

Skill/agent:
`$neuroforge-training-loop`

Constraints:
Reject empty parameter lists and invalid hyperparameters. Keep optimizer state tied to parameter order. Do not add global state.

Expected output:
Two new optimizer classes that work with existing models.

Tests required:
Known MomentumSGD update, known Adam first-step update, zeroGrad behavior, invalid hyperparameters throw, tiny regression loss decreases.

Definition of done:
Optimizers pass unit tests and can train a small deterministic model.

Risks:
Adam bias correction is easy to implement incorrectly.

## Ticket 3

Task:
Add activation and classification helpers.

Context:
Current activations are ReLU, Sigmoid, and Tanh. Multi-class demos need softmax-style output and accuracy helpers.

Goal:
Add `LeakyReLU`, `Softmax`, and `CrossEntropyLoss` for rank-2 batch inputs.

Files likely involved:
`include/neuroforge/nn`, `src/nn`, `include/neuroforge/losses`, `src/losses`, `tests`

Dependencies:
Stable Tensor operations and loss infrastructure.

Branch:
`model-features`

Skill/agent:
`$neuroforge-nn-layers`, `$neuroforge-autograd-engine`

Constraints:
Use numerically stable softmax by subtracting row max. Support rank-2 logits `[batch, classes]`. Targets should use one-hot rank-2 tensors for the first version.

Expected output:
Users can build small multi-class dense classifiers.

Tests required:
LeakyReLU forward/backward, softmax rows sum to 1, cross entropy known value, gradient shape, invalid target shape throws.

Definition of done:
Multi-class loss and activation tests pass and docs explain target format.

Risks:
Softmax plus cross entropy can duplicate gradient logic if not designed carefully.

## Ticket 4

Task:
Add Trainer/DataLoader integration.

Context:
`DataLoader` exists, but `Trainer` currently trains full tensors directly.

Goal:
Add `Trainer::fit(DataLoader&, const TrainingConfig&)` or equivalent batch training API.

Files likely involved:
`include/neuroforge/training`, `src/training`, `tests`

Dependencies:
Existing `DataLoader`, `Batch`, `Trainer`, losses, optimizers.

Branch:
`model-features`

Skill/agent:
`$neuroforge-training-loop`

Constraints:
Keep existing Tensor-based `fit` APIs. Record average epoch loss. Do not make `Trainer` own datasets.

Expected output:
Mini-batch training can use existing `DataLoader`.

Tests required:
Batch training changes parameters, average loss is recorded, deterministic shuffle works, evaluation does not update parameters.

Definition of done:
Tensor training tests and DataLoader training tests both pass.

Risks:
Changing training semantics can break existing examples if overloads are not kept separate.

## Ticket 5

Task:
Add regularization and mode-aware layers.

Context:
Dropout and BatchNorm need train/eval behavior. NeuroForge currently has no mode concept.

Goal:
Add `Module::train()`, `Module::eval()`, `Dropout`, and `BatchNorm1d`.

Files likely involved:
`include/neuroforge/nn`, `src/nn`, `tests`

Dependencies:
Stable `Module` API and Tensor ops.

Branch:
`model-features`

Skill/agent:
`$neuroforge-nn-layers`, `$neuroforge-training-loop`

Constraints:
Preserve existing modules. Dropout should be deterministic in tests through a seed path. BatchNorm1d supports rank-2 `[batch, features]` only.

Expected output:
Users can build basic dense networks with regularization.

Tests required:
Train/eval propagation through `Sequential`, Dropout train/eval behavior, BatchNorm shape and running statistics, no parameter loss.

Definition of done:
Mode-aware modules work and are documented.

Risks:
BatchNorm adds state and can make tests flaky if random inputs are not controlled.

## Ticket 6

Task:
Add examples and docs for Phase 9.

Context:
New features need proof surfaces.

Goal:
Add a small binary classification or multi-class dense example and update docs.

Files likely involved:
`examples`, `data`, `docs`, `README.md`

Dependencies:
Tickets 1 through 5 as available.

Branch:
`model-features`

Skill/agent:
`$neuroforge-training-loop`, documentation agent

Constraints:
No fake predictions. Keep examples deterministic and short.

Expected output:
One new example showing non-MSE/non-SGD training.

Tests required:
Example builds, test suite passes, documented commands work.

Definition of done:
Docs clearly list supported losses, optimizers, activations, and model limits.

Risks:
Too many new examples can make the project noisy.

## CNN Decision

Do not add CNNs in Phase 9.

CNN support is a separate major architecture phase because it needs:

- N-D `Shape`
- N-D `Tensor` indexing
- reshape and flatten
- convolution forward
- convolution backward
- pooling
- image-like datasets
- many new gradient tests

Treat CNN work as Phase 11 or later.

## Phase 9 Definition Of Done

- New losses pass tests.
- New optimizers pass tests.
- Dense classification helpers pass tests.
- DataLoader training works without breaking Tensor training.
- Mode-aware layers work if included.
- Docs and README accurately list supported features.
- Default build, examples, and tests pass.
