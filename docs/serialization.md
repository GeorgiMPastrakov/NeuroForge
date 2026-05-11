# Serialization

NeuroForge provides simple text model serialization for `Sequential`.

## Supported Models

Serialization supports `Sequential` models containing:

- `Linear`
- `ReLU`
- `Sigmoid`
- `Tanh`

`Linear` stores full weight and bias values.

Activation layers store their layer type.

## Saving

Use:

```cpp
ModelSaver::save(model, path);
```

The saved format is readable text. It is meant for demos and inspection, not compact production storage.

## Loading

Use:

```cpp
Sequential loaded = ModelLoader::load(path);
```

Loaded models should produce the same predictions as the saved model for the supported layer set.

## Limits

Serialization does not support:

- optimizer state
- arbitrary custom modules
- training history
- binary checkpoints
- versioned migration
