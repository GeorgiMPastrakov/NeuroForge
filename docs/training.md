# Training

The training module connects models, losses, optimizers, and history.

## Main Types

- `TrainingConfig`
- `TrainingHistory`
- `Trainer`

## TrainingConfig

`TrainingConfig` stores simple loop settings:

- epochs
- log interval
- verbose flag

Learning rate is configured on the optimizer.

## TrainingHistory

`TrainingHistory` stores epoch numbers and loss values.

It is used by:

- tests
- examples
- Visual Lab loss plots

## Manual Training

`Trainer::fit` uses the manual backward path:

```text
optimizer.zeroGrad
model.forward
loss.forward
loss.backward
model.backward
optimizer.step
history.addLoss
```

This path is used by the XOR, regression, and dense classification examples.

## Autograd Training

`Trainer::fitAutograd` uses Tensor autograd:

```text
optimizer.zeroGrad
model.enableAutograd
model.forward
loss.forward
loss.backward
model.syncAutogradGradients
optimizer.step
history.addLoss
```

## Evaluation

`Trainer::evaluateLoss` runs forward and loss calculation only.

It does not call backward and does not update parameters.

## DataLoader Training

`Trainer` can train directly from `DataLoader` batches.

Supported overloads:

- `fit(const DataLoader&, const TrainingConfig&)`
- `fitAutograd(const DataLoader&, const TrainingConfig&)`
- `evaluateLoss(const DataLoader&)`

Epoch loss is stored as the row-weighted average of batch losses.

Training calls put the model in train mode. Evaluation temporarily uses eval mode and restores the previous model mode.

## Train And Eval Mode

`Module` supports:

- `train`
- `eval`
- `isTraining`

`Sequential` propagates train/eval mode to child layers.

`Dropout` uses train/eval mode. Standard deterministic layers behave the same in both modes.

## Current Limits

The Trainer supports full Tensor inputs and materialized DataLoader batches. It does not yet own datasets or stream infinite data sources.
