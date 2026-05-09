# Phase 6: Data, Metrics, Serialization

## Goal

Add practical dataset handling, evaluation metrics, and model persistence after the training MVP works.

## Ticket 1

Task:
Implement dataset and CSV loading.

Context:
Regression demos need numeric data loaded from files.

Goal:
Add `Dataset` and `CSVDataset` for numeric CSV with a configurable label column.

Files likely involved:
`include/neuroforge/data/Dataset.hpp`, `CSVDataset.hpp`, `src/data`, tests

Dependencies:
Phase 4.

Branch:
`data-metrics-serialization`

Skill/agent:
Data agent.

Constraints:
Strict numeric parsing. Clear errors for invalid rows and missing files.

Expected output:
CSV files load into feature and label tensors.

Tests required:
Valid CSV, missing file, invalid numeric value, label-column selection.

Definition of done:
Small test CSV loads deterministically.

Risks:
Loose parsing will hide bad dataset errors.

## Ticket 2

Task:
Implement `DataLoader` and train/test split.

Context:
Training should support batches and evaluation splits.

Goal:
Add `Batch`, `DataLoader`, and `TrainTestSplitter`.

Files likely involved:
`include/neuroforge/data/DataLoader.hpp`, `TrainTestSplitter.hpp`, `src/data`, tests

Dependencies:
Ticket 1.

Branch:
`data-metrics-serialization`

Skill/agent:
Data agent.

Constraints:
Keep batching simple. Shuffle must be deterministic with seed.

Expected output:
Dataset can produce batches and train/test splits.

Tests required:
Batch count, batch shape, deterministic shuffle, split sizes.

Definition of done:
Trainer can train from loader or tensors through a clear API.

Risks:
Overbuilding iterators can delay useful demos.

## Ticket 3

Task:
Implement metrics.

Context:
Loss is not enough to explain model quality.

Goal:
Add MSE, MAE, Accuracy, and BinaryAccuracy.

Files likely involved:
`include/neuroforge/metrics`, `src/metrics`, tests

Dependencies:
Phase 4.

Branch:
`data-metrics-serialization`

Skill/agent:
Training or metrics agent.

Constraints:
BinaryAccuracy threshold is `0.5`.

Expected output:
Metrics work for regression and binary classification.

Tests required:
Known MSE, MAE, binary accuracy, shape mismatch throws.

Definition of done:
Metrics can be used by trainer evaluation.

Risks:
Metric shape rules must match loss shape rules.

## Ticket 4

Task:
Implement model save/load.

Context:
Users should be able to persist trained models.

Goal:
Add simple text serialization for `Sequential` with `Linear` weights and biases.

Files likely involved:
`include/neuroforge/serialization/ModelSaver.hpp`, `ModelLoader.hpp`, `src/serialization`, tests

Dependencies:
Phase 4.

Branch:
`data-metrics-serialization`

Skill/agent:
Serialization agent.

Constraints:
Use a simple readable text format. Do not add binary format.

Expected output:
Save and load preserve model parameters.

Tests required:
Save model, load model, predictions match before and after load, malformed file throws.

Definition of done:
Persistence works for MVP `Sequential` models.

Risks:
Loading arbitrary architecture can get complex. Start with supported layer types only.

## Ticket 5

Task:
Add CSV regression example.

Context:
The project needs a second demo beyond XOR.

Goal:
Train a small regression model from numeric CSV and report MSE.

Files likely involved:
`examples/linear_regression.cpp`, `data/linear_regression.csv`, tests or manual check

Dependencies:
Tickets 1 through 4.

Branch:
`data-metrics-serialization`

Skill/agent:
Data and training agent.

Constraints:
Use real training. Do not hardcode predictions.

Expected output:
Regression example loads CSV, trains, evaluates, and saves model.

Tests required:
Example builds. Optional integration test proves loss decreases on tiny regression data.

Definition of done:
Regression demo works through the public API.

Risks:
Feature scaling may be needed for stable training.
