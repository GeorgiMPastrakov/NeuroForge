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

This path is used by the XOR and regression examples.

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

## Current Limits

The Trainer currently trains full Tensor inputs directly. DataLoader integration can be added later, but current demos keep training explicit and small.
