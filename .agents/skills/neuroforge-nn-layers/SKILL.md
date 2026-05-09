---
name: neuroforge-nn-layers
description: Use this skill when implementing or reviewing NeuroForge neural network components such as Parameter, Module, Linear, ReLU, Sigmoid, Tanh, Sequential, model.summary, parameter collection, forward passes, and layer-level backward logic.
---

# NeuroForge Neural Network Layers Skill

## Purpose

You are the Neural Network Layers specialist for NeuroForge.

Your job is to build the model API of the framework.

This module should make NeuroForge feel like a small C++ version of `torch.nn`.

The goal is to let users define models using readable building blocks like:

```cpp
Sequential model;
model.add(std::make_shared<Linear>(2, 4));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(4, 1));
model.add(std::make_shared<Sigmoid>());
```

---

## When To Use This Skill

Use this skill when the task involves:

* `Parameter`
* `Module`
* `Linear`
* `Dense`
* `ReLU`
* `Sigmoid`
* `Tanh`
* `Sequential`
* model architecture
* forward pass through layers
* parameter collection
* model summary
* layer names
* layer shapes
* layer backward pass if manual backprop is used

Do not use this skill for:

* low-level tensor implementation
* optimizer update rules
* full training loop orchestration
* GUI rendering
* CSV parsing

---

## Main Goal

The Neural Network module should provide a clean API for building simple feedforward neural networks.

Minimum target:

```text
Linear
ReLU
Sigmoid
Tanh
Sequential
Parameter collection
Forward pass
Backward support through either autograd or manual backward
Model summary
```

---

## Design Principles

Follow these principles:

1. Keep the API simple.
2. Make layers composable.
3. Use `std::shared_ptr<Module>` for storing layers.
4. Avoid raw owning pointers.
5. Every trainable value must be exposed through `parameters()`.
6. Shape errors must be clear.
7. Layers should have readable names.
8. Sequential should forward input through layers in order.
9. The code should be understandable to someone learning PyTorch internals.

---

## Main Classes

The module may include:

```text
Parameter
Module
Linear
Activation
ReLU
Sigmoid
Tanh
Sequential
ModelSummary
```

---

## Parameter

A `Parameter` represents a trainable tensor.

It should contain:

```text
Tensor value
Tensor gradient or grad storage
requires_grad = true
optional name
```

Possible API:

```cpp
class Parameter {
public:
    Tensor& data();
    const Tensor& data() const;

    Tensor& grad();
    const Tensor& grad() const;

    void zero_grad();
    std::string name() const;
};
```

The optimizer will update `Parameter` objects.

---

## Module Base Class

Suggested interface:

```cpp
class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;

    virtual std::vector<Parameter*> parameters() {
        return {};
    }

    virtual void zero_grad() {
        for (Parameter* parameter : parameters()) {
            parameter->zero_grad();
        }
    }

    virtual std::string name() const = 0;

    virtual ~Module() = default;
};
```

If manual backprop is used before full autograd, the interface may include:

```cpp
virtual Tensor backward(const Tensor& grad_output);
```

Only add this if the architecture needs it.

---

## Linear Layer

`Linear` is the most important layer.

It represents:

```text
y = xW + b
```

Suggested constructor:

```cpp
Linear(size_t in_features, size_t out_features);
```

It should store:

```text
weights
bias
in_features
out_features
cached input if manual backward is used
```

Shape rules:

```text
input shape:  [batch_size, in_features]
weight shape: [in_features, out_features]
bias shape:   [1, out_features] or [out_features]
output shape: [batch_size, out_features]
```

If bias broadcasting is not implemented, handle bias addition explicitly.

---

## Linear Initialization

Use small random values.

Acceptable first approach:

```text
random uniform values around 0
```

Better later:

```text
Xavier/Glorot initialization
He initialization for ReLU networks
```

Do not overcomplicate initialization before training works.

---

## Activation Layers

Implement:

```text
ReLU
Sigmoid
Tanh
```

Each activation should:

```text
take a Tensor
return a Tensor of the same shape
support backward through autograd or manual derivative
have no trainable parameters
```

### ReLU

```text
relu(x) = max(0, x)
```

### Sigmoid

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

### Tanh

```text
tanh(x)
```

---

## Sequential

`Sequential` stores layers and runs them in order.

Suggested API:

```cpp
class Sequential : public Module {
public:
    void add(std::shared_ptr<Module> module);

    Tensor forward(const Tensor& input) override;

    std::vector<Parameter*> parameters() override;

    std::string name() const override;

    std::string summary() const;
};
```

Forward rule:

```text
output = input
for each layer:
    output = layer.forward(output)
return output
```

Parameter rule:

```text
collect parameters from all child modules
```

---

## Model Summary

Implement a readable model summary.

Example:

```text
Sequential(
  [0] Linear(2 -> 4), parameters: 12
  [1] ReLU()
  [2] Linear(4 -> 1), parameters: 5
  [3] Sigmoid()
)

Total parameters: 17
```

Parameter count:

```text
Linear(in, out) = in*out + out
```

This summary will also support the visualization module.

---

## Required Tests

Add or update tests for:

```text
Parameter stores data and gradient
Parameter zero_grad works
Linear forward shape
Linear parameter count
Linear rejects invalid input shape
ReLU output values
Sigmoid output values
Tanh output values
Sequential forward shape
Sequential collects parameters from all layers
Sequential summary contains layer names
XOR model can be constructed
```

If manual backward exists, also test:

```text
Linear backward gradient shapes
Activation backward outputs
Sequential backward order
```

---

## Definition Of Done

A Neural Network task is done when:

```text
The layer/module API is clean.
Forward pass works.
Shapes are validated.
Parameters are exposed correctly.
Tests cover expected behavior.
The implementation supports the training loop.
The implementation supports future visualization.
```

---

## Review Checklist

Before finishing, check:

```text
Does every trainable layer expose parameters?
Does Sequential collect parameters recursively?
Are layer names readable?
Do shape errors explain the problem?
Can this layer be used in the XOR model?
Does this work with the current Tensor design?
Does this work with either manual backprop or autograd path?
```

---

## Important Warning

Do not make the layer system depend only on XOR.

The XOR demo is important, but the API should also support future regression and 2D classification demos.

Do not hardcode layer sizes, dataset sizes, or prediction behavior.
