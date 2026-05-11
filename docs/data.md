# Data And Metrics

NeuroForge includes a small practical data layer for numeric examples.

## Dataset

`Dataset` owns feature and label tensors.

Rules:

- features must be rank 2
- labels must be rank 2
- row counts must match

Supported behavior:

- size
- feature access
- label access
- single sample lookup
- subset by index list

## CSV Loading

`CSVDataset` loads strict numeric CSV files.

Supported behavior:

- optional header row
- selected label column
- all other columns become features
- labels are shaped as `[rows, 1]`

Invalid rows, missing files, bad numbers, and invalid label columns throw exceptions.

## DataLoader

`DataLoader` materializes batches from a `Dataset`.

Supported behavior:

- fixed batch size
- optional deterministic shuffle
- local random seed

It returns a vector of `Batch` values for simplicity.

## Train/Test Split

`TrainTestSplitter` splits a dataset by ratio.

It supports deterministic shuffling with a local seed.

## Metrics

Supported metric functions:

- MSE
- MAE
- accuracy
- binary accuracy

Binary accuracy uses threshold `0.5`.

All metrics require exact shape matches.
