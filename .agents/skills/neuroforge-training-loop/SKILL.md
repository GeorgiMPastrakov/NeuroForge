---
name: neuroforge-training-loop
description: Use this skill when implementing or reviewing NeuroForge's Trainer, TrainingConfig, TrainingHistory, epoch loops, batch loops, optimizer integration, loss tracking, metrics, evaluation, and training demos such as XOR or CSV regression.
---

# NeuroForge Training Loop Skill

## Purpose

You are the Training Loop specialist for NeuroForge.

Your job is to make models actually learn.

This skill focuses on the system that connects models, losses, optimizers, datasets, metrics, logging, and training history.

The training loop is where NeuroForge proves it is a real machine learning framework, not just a set of classes.

---

## When To Use This Skill

Use this skill when the task involves:

- `Trainer`
- `TrainingConfig`
- `TrainingHistory`
- epoch loop
- batch loop
- loss tracking
- accuracy tracking
- optimizer integration
- `zero_grad`
- `backward`
- `step`
- model evaluation
- XOR demo
- regression demo
- training logs
- convergence debugging

Do not use this skill for:

- low-level matrix multiplication
- layer implementation details, except integration
- GUI rendering, except exposing training history
- CSV parsing internals

---

## Main Goal

The training system should allow simple usage:

```cpp
MSELoss loss;
SGD optimizer(model.parameters(), 0.1);

Trainer trainer(model, loss, optimizer);

TrainingConfig config;
config.epochs = 1000;
config.log_interval = 100;

TrainingHistory history = trainer.fit(dataset, config);
```

It should produce useful information:

```text
Epoch 1: loss = 0.693
Epoch 100: loss = 0.421
Epoch 500: loss = 0.083
Epoch 1000: loss = 0.018
```

---

## Core Training Process

A standard training step should follow this structure:

```text
1. Get input and target.
2. Run model.forward(input).
3. Compute loss.
4. Clear old gradients.
5. Run backward pass.
6. Optimizer updates parameters.
7. Store loss and metrics.
```

Typical order:

```cpp
optimizer.zero_grad();
Tensor prediction = model.forward(input);
Tensor loss = loss_fn.forward(prediction, target);
loss.backward();
optimizer.step();
```

If manual backprop is used, adapt the order but keep the same conceptual process.

---

## Main Classes

The training module may include:

```text
Trainer
TrainingConfig
TrainingHistory
TrainingResult
ExperimentLogger
EarlyStopping
```

Start with:

```text
Trainer
TrainingConfig
TrainingHistory
```

---

## TrainingConfig

`TrainingConfig` should store configuration.

Suggested fields:

```cpp
struct TrainingConfig {
    size_t epochs = 1000;
    size_t batch_size = 1;
    bool shuffle = true;
    size_t log_interval = 100;
    bool verbose = true;
};
```

Learning rate usually belongs to the optimizer, not the config, unless the project chooses otherwise.

---

## TrainingHistory

`TrainingHistory` stores values needed for logs and visualization.

Suggested fields:

```text
epoch numbers
training loss per epoch
validation loss per epoch
training accuracy per epoch
gradient norms per epoch
```

Minimum:

```text
loss per epoch
```

This class should be usable by the visualization module.

Example API:

```cpp
history.add_loss(epoch, loss_value);
history.losses();
history.final_loss();
history.print();
```

---

## Trainer Responsibilities

`Trainer` should:

```text
run the training loop
call model.forward
call loss function
call backward
call optimizer.step
call optimizer.zero_grad
store TrainingHistory
evaluate model
optionally calculate metrics
```

The trainer should not:

```text
own tensor math logic
own GUI rendering code
own CSV parsing logic
hardcode XOR
hardcode model architecture
```

---

## Evaluation

Add evaluation separately from training.

Example:

```cpp
EvaluationResult result = trainer.evaluate(test_data);
```

Evaluation should:

```text
run forward pass
calculate loss
calculate metrics
not update parameters
not call optimizer.step
```

---

## Metrics

Training may use:

```text
MSE
MAE
Accuracy
BinaryAccuracy
```

For XOR:

```text
prediction >= 0.5 => class 1
prediction < 0.5 => class 0
```

---

## XOR Training Requirements

The XOR demo should show real learning.

Dataset:

```text
[0, 0] -> 0
[0, 1] -> 1
[1, 0] -> 1
[1, 1] -> 0
```

Suggested model:

```text
Linear(2 -> 4)
Sigmoid or ReLU
Linear(4 -> 1)
Sigmoid
```

Suggested training:

```text
Loss: MSE or BCE
Optimizer: SGD
Epochs: enough to decrease loss
Seed: deterministic
```

Expected final behavior:

```text
[0, 0] prediction near 0
[0, 1] prediction near 1
[1, 0] prediction near 1
[1, 1] prediction near 0
```

Do not hardcode predictions.

---

## Regression Demo Requirements

The CSV regression demo should show:

```text
CSV loading
features/labels
train/test split
model training
MSE evaluation
prediction for new input
```

The training loop should not care whether data came from CSV or hardcoded tensors.

---

## Required Tests

Add or update tests for:

```text
TrainingConfig defaults
TrainingHistory stores losses
Trainer can run one epoch
Trainer decreases loss on simple regression
Trainer decreases loss on XOR or a tiny classification task
Evaluation does not update parameters
optimizer.zero_grad is called correctly
TrainingHistory final loss is available
```

Use deterministic seeds.

Avoid flaky tests.

---

## Debugging Checklist

If training does not work, check:

```text
Are weights initialized randomly, not all the same?
Are activations correct?
Is loss derivative correct?
Are gradients zeroed at the right time?
Are gradients accumulating accidentally?
Is learning rate too high?
Is learning rate too low?
Are target shapes correct?
Are prediction shapes correct?
Is Sigmoid saturating?
Is the model too small?
```

---

## Definition Of Done

A Training task is done when:

```text
The training loop runs.
Loss is tracked.
Parameters are updated.
Tests prove loss can decrease.
Evaluation exists or is clearly planned.
TrainingHistory can support visualization.
No demo result is faked.
The code is not hardcoded for one dataset.
```

---

## Review Checklist

Before finishing, check:

```text
Does Trainer depend on clean interfaces?
Does it work with Sequential?
Does it work with the current LossFunction design?
Does it call zero_grad, backward, and step in the correct order?
Does TrainingHistory store enough data for the visualizer?
Can the same Trainer train XOR and regression?
Are logs readable?
```

---

## Important Warning

Do not make the Trainer too magical.

Keep it simple and explicit.

The user should still be able to understand:

```text
forward
loss
backward
optimizer step
```

That educational clarity is more important than advanced automation.
