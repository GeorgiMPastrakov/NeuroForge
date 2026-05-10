# AGENTS.md — NeuroForge

## 0. User Preferences

- Do not name branches with `codex/` in front.
- Use absolutely zero comments in code.

## 1. Project Identity

You are working on **NeuroForge**, a C++ machine learning framework inspired by PyTorch.

NeuroForge is not just one AI model. It is a small educational machine learning framework that allows users to create, train, evaluate, save, load, and visualize simple ML models.

The project should demonstrate how machine learning frameworks work internally:

- Tensor and matrix operations
- Matrix multiplication
- Computational graphs
- Autograd
- Backpropagation
- Neural network layers
- Loss functions
- Optimizers
- Dataset loading
- Training loops
- Metrics
- Model saving/loading
- Visualization of architecture and training

The project should feel like a **small CPU-only PyTorch-like framework written in C++**.

---

## 2. Main Agent Role

Act as the **technical CEO, architect, and orchestrator** of NeuroForge.

Your job is not only to write code. Your job is to protect the long-term architecture of the project.

You must:

- Keep the project coherent.
- Split large work into smaller tasks.
- Decide implementation order.
- Keep the code clean and testable.
- Avoid random features that do not fit the project.
- Review changes critically.
- Prefer simple working systems over unfinished complex systems.
- Think about how every feature fits the full framework.

Before implementing any large task, think:

- Which module owns this?
- Which classes are affected?
- What is the smallest correct version?
- What tests prove this works?
- Will this make the project cleaner or messier?
- Does this help NeuroForge become a better ML framework?

---

## 3. What NeuroForge Is

NeuroForge is:

- A C++ machine learning framework.
- A small educational version of PyTorch.
- CPU-only.
- Built from scratch.
- Focused on understanding ML internals.
- Designed with many meaningful classes.
- Suitable for training small models such as XOR, linear regression, and 2D classification.
- A project that demonstrates C++ OOP, linear algebra, backpropagation, and ML architecture.

---

## 4. What NeuroForge Is Not

NeuroForge is not:

- A real replacement for PyTorch.
- A CUDA/GPU framework.
- A production ML system.
- A transformer framework.
- A wrapper around PyTorch, TensorFlow, or another ML library.
- A project that hides the math inside external ML libraries.

Do not use external machine learning libraries for the core framework.

External libraries are acceptable only for:

- Testing
- GUI/visualization
- Build support
- Simple file parsing if needed

The machine learning logic must be implemented by NeuroForge.

---

## 5. Core Goal

The final framework should allow code that feels like this:

```cpp
#include "neuroforge/neuroforge.hpp"

using namespace neuroforge;

int main() {
    Tensor X = Tensor::fromVector({
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    });

    Tensor y = Tensor::fromVector({
        {0},
        {1},
        {1},
        {0}
    });

    Sequential model;

    model.add(std::make_shared<Linear>(2, 4));
    model.add(std::make_shared<Sigmoid>());
    model.add(std::make_shared<Linear>(4, 1));
    model.add(std::make_shared<Sigmoid>());

    MSELoss loss;
    SGD optimizer(model.parameters(), 0.1);

    Trainer trainer(model, loss, optimizer);
    trainer.fit(X, y, 1000);

    Tensor prediction = model.forward(X);
    prediction.print();

    return 0;
}
```

The goal is to show this process:

```text
input data
→ tensor operations
→ forward pass
→ loss calculation
→ backward pass
→ gradients
→ optimizer update
→ improved model
```

---

## 6. Engineering Principles

Follow these principles at all times:

1. Correctness first.
2. Clear architecture before features.
3. Small working increments.
4. Tests for core math.
5. Tests for training behavior.
6. No fake training results.
7. No hardcoded demo predictions.
8. Avoid overengineering.
9. Avoid global state unless truly needed.
10. Use modern C++ ownership rules.
11. Prefer RAII.
12. Prefer `std::vector`, `std::shared_ptr`, and `std::unique_ptr`.
13. Avoid raw owning pointers.
14. Keep APIs simple.
15. Make shape errors obvious.
16. Every class must have a clear reason to exist.
17. Every feature must support the NeuroForge vision.

---

## 7. C++ Style

Use modern C++.

Preferred standard:

```text
C++17 or C++20
```

Use:

```cpp
std::vector
std::string
std::shared_ptr
std::unique_ptr
std::optional
std::runtime_error
std::invalid_argument
```

Avoid:

```cpp
raw owning pointers
manual new/delete unless absolutely necessary
macros for normal logic
unnecessary templates
large header-only messes
hidden magic
```

Use namespace:

```cpp
namespace neuroforge {
}
```

Do not add comments to code.

---

## 8. Project Structure

Use this structure:

```text
NeuroForge/
│
├── CMakeLists.txt
├── AGENTS.md
├── README.md
│
├── include/
│   └── neuroforge/
│       ├── core/
│       ├── autograd/
│       ├── nn/
│       ├── losses/
│       ├── optim/
│       ├── data/
│       ├── training/
│       ├── metrics/
│       ├── serialization/
│       └── visualization/
│
├── src/
│   ├── core/
│   ├── autograd/
│   ├── nn/
│   ├── losses/
│   ├── optim/
│   ├── data/
│   ├── training/
│   ├── metrics/
│   ├── serialization/
│   └── visualization/
│
├── tests/
├── examples/
├── data/
├── docs/
└── tools/
```

Build commands should be:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

---

## 9. Main Modules

NeuroForge should be organized into these modules:

```text
Core
Autograd
Neural Network
Losses
Optimizers
Data
Training
Metrics
Serialization
Visualization
Examples
Tests
Documentation
```

Each module must have a clear responsibility.

---

## 10. Core Module

The Core module contains mathematical foundations.

Main classes:

```text
Tensor
Matrix
Shape
Random
Initializer
```

### Tensor

`Tensor` is the central data object.

It should support:

* Data storage
* Shape
* Indexing
* 1D and 2D tensors first
* Reshape
* Transpose for 2D tensors
* Element-wise operations
* Matrix multiplication
* Optional gradient storage
* Optional autograd tracking

Suggested fields:

```cpp
std::vector<double> data_;
std::vector<size_t> shape_;
std::vector<double> grad_;
bool requires_grad_;
```

Do not attempt full PyTorch broadcasting at the beginning.

Start simple.

### Matrix

`Matrix` may exist as a simpler 2D class.

If both `Matrix` and `Tensor` exist, define their relationship clearly.

Preferred approach:

* `Tensor` is the main long-term object.
* `Matrix` can be an early helper or wrapper for 2D operations.
* Avoid duplicated logic long-term.

### Shape

`Shape` should help with dimension validation.

It can provide:

* Rank
* Rows
* Columns
* Total element count
* Shape comparison
* Pretty printing

### Core Operations

Minimum operations:

```text
add
subtract
element-wise multiply
scalar multiply
matmul
transpose
sum
mean
pow
exp
log
relu
sigmoid
tanh
```

Prioritize the operations needed for XOR and regression.

---

## 11. Autograd Module

The Autograd module implements automatic differentiation.

The goal is to support this style:

```cpp
Tensor prediction = model.forward(input);
Tensor loss = mse(prediction, target);
loss.backward();
```

After `backward()`, trainable parameters should contain gradients.

### Autograd Concepts

The autograd system should represent:

```text
tensor data
gradient
requires_grad
operation that created the tensor
parents of the tensor
backward function
topological order
```

Possible classes:

```text
ComputationGraph
Operation
AddOperation
SubOperation
MulOperation
MatMulOperation
ReLUOperation
SigmoidOperation
TanhOperation
MSEOperation
BackwardEngine
```

### Important Gradient Rules

Addition:

```text
z = x + y
dz/dx = 1
dz/dy = 1
```

Multiplication:

```text
z = x * y
dz/dx = y
dz/dy = x
```

Matrix multiplication:

```text
Z = XW
dL/dX = dL/dZ * W^T
dL/dW = X^T * dL/dZ
```

ReLU:

```text
relu(x) = max(0, x)
gradient = 1 if x > 0 else 0
```

Sigmoid:

```text
sigmoid(x) = 1 / (1 + exp(-x))
gradient = sigmoid(x) * (1 - sigmoid(x))
```

### Autograd Strategy

Do not start by trying to build full PyTorch autograd.

Recommended path:

1. Build `Tensor` and matrix operations.
2. Build manual backprop for `Linear` layers.
3. Build a small scalar `Value` autograd engine for learning/testing.
4. Extend toward tensor autograd.
5. Connect tensor autograd to model training once stable.

Correct training is more important than pretending to support full autograd.

---

## 12. Neural Network Module

This module provides the model-building API.

Main classes:

```text
Parameter
Module
Linear
Activation
ReLU
Sigmoid
Tanh
Sequential
```

### Parameter

A `Parameter` is a trainable tensor.

It should contain:

* Value
* Gradient
* `requires_grad = true`
* Optional name

Parameters are updated by optimizers.

### Module

`Module` is the base class for neural network components.

Suggested interface:

```cpp
class Module {
public:
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Parameter*> parameters() = 0;
    virtual void zero_grad();
    virtual std::string name() const = 0;
    virtual ~Module() = default;
};
```

### Linear Layer

`Linear` is the most important layer.

It represents:

```text
y = xW + b
```

It should store:

```text
weights
bias
cached input for backward
gradients
```

Constructor:

```cpp
Linear(size_t in_features, size_t out_features);
```

### Activation Layers

Implement:

```text
ReLU
Sigmoid
Tanh
```

They should work as modules so they can be used inside `Sequential`.

### Sequential

`Sequential` stores layers in order.

Example:

```cpp
Sequential model;
model.add(std::make_shared<Linear>(2, 4));
model.add(std::make_shared<ReLU>());
model.add(std::make_shared<Linear>(4, 1));
model.add(std::make_shared<Sigmoid>());
```

It should support:

```text
forward
parameters
summary
save/load eventually
```

---

## 13. Loss Module

Loss functions compare predictions with targets.

Main classes:

```text
LossFunction
MSELoss
MAELoss
BinaryCrossEntropyLoss
```

### LossFunction

Suggested interface:

```cpp
class LossFunction {
public:
    virtual Tensor forward(const Tensor& prediction, const Tensor& target) = 0;
    virtual Tensor backward() = 0;
    virtual std::string name() const = 0;
    virtual ~LossFunction() = default;
};
```

If tensor autograd is active, the loss may produce a tensor and rely on `loss.backward()`.

### Required Losses

Minimum:

```text
MSELoss
```

Good extras:

```text
MAELoss
BinaryCrossEntropyLoss
```

Use MSE first because it is simple and works for XOR/regression demos.

---

## 14. Optimizer Module

Optimizers update parameters using gradients.

Main classes:

```text
Optimizer
SGD
MomentumSGD
Adam
```

### Optimizer

Suggested interface:

```cpp
class Optimizer {
public:
    virtual void step() = 0;
    virtual void zero_grad() = 0;
    virtual ~Optimizer() = default;
};
```

### SGD

Implement this first.

Formula:

```text
parameter = parameter - learning_rate * gradient
```

### MomentumSGD

Optional second optimizer.

Formula:

```text
velocity = momentum * velocity - learning_rate * gradient
parameter = parameter + velocity
```

### Adam

Advanced optional optimizer.

Do not implement Adam before simple SGD training works.

---

## 15. Data Module

This module connects NeuroForge to practical machine learning.

Main classes:

```text
Dataset
CSVDataset
Batch
DataLoader
TrainTestSplitter
Preprocessor
```

### Dataset

A dataset stores features and labels.

It should support:

```text
size
get sample
features
labels
```

### CSVDataset

Loads numeric CSV data.

Basic functionality:

```text
read file
parse numeric values
choose label column
store features and labels
handle invalid rows
```

### DataLoader

Provides batches.

It should support:

```text
batch size
shuffle
iteration over batches
```

Simple implementation is acceptable.

### TrainTestSplitter

Splits dataset:

```text
80% train
20% test
```

### Preprocessor

Useful preprocessing:

```text
normalize columns
standardize columns
handle missing values
```

Only implement what is needed for demos.

---

## 16. Training Module

The Training module manages the training loop.

Main classes:

```text
Trainer
TrainingConfig
TrainingHistory
ExperimentLogger
```

### Trainer

`Trainer` owns the high-level loop.

Responsibilities:

```text
forward pass
loss calculation
backward pass
optimizer step
zero gradients
metric calculation
history logging
optional visualization updates
```

Example:

```cpp
Trainer trainer(model, loss, optimizer);
trainer.fit(train_loader, epochs);
trainer.evaluate(test_loader);
```

### TrainingConfig

Stores:

```text
epochs
learning rate
batch size
shuffle
log interval
```

### TrainingHistory

Stores:

```text
loss per epoch
accuracy per epoch
gradient norms
```

### ExperimentLogger

Writes experiment results to files:

```text
experiment name
model architecture
dataset
epochs
learning rate
final loss
final accuracy
```

---

## 17. Metrics Module

Metrics evaluate model quality.

Main metrics:

```text
MSE
MAE
Accuracy
BinaryAccuracy
```

For XOR:

```text
prediction >= 0.5 means class 1
prediction < 0.5 means class 0
```

For regression:

```text
MSE and MAE are enough
```

---

## 18. Serialization Module

This module saves and loads models.

Main classes:

```text
ModelSaver
ModelLoader
Checkpoint
```

Minimum behavior:

```text
save layer names
save weights
save biases
load weights
load biases
```

Use a simple text format first.

Example:

```text
Linear 2 4
weights:
0.1 0.2 0.3 0.4
...
bias:
0.0 0.0 0.0 0.0
```

Do not build a complex binary format at the beginning.

---

## 19. Visualization Module

NeuroForge should include a graphical visualization tool called:

```text
NeuroForge Visual Lab
```

The visualizer should be optional and separate from the core framework.

The core ML library must work without the GUI.

Preferred technology:

```text
Dear ImGui + ImPlot
```

Main classes:

```text
VisualApp
ModelGraphView
LossPlotView
GradientView
TensorInspectorView
DatasetScatterView
DecisionBoundaryView
```

### Visual Feature 1: Model Architecture View

Show the model visually as connected blocks:

```text
Input → Linear(2→4) → ReLU → Linear(4→1) → Sigmoid → Output
```

Each block should display:

```text
layer name
input shape
output shape
parameter count
```

### Visual Feature 2: Training Loss Plot

Show loss over epochs as a live graph.

This is one of the most important visual features.

### Visual Feature 3: Gradient Monitor

Show gradient norms per layer.

Example:

```text
Linear1 weights grad norm: 0.042
Linear2 weights grad norm: 0.017
```

This helps prove that backpropagation is working.

### Visual Feature 4: Tensor Inspector

Allow inspecting tensor values.

Example:

```text
Tensor: Linear1.weights
Shape: [2 x 4]
Values:
0.21 -0.43 0.12 0.88
0.10  0.56 -0.31 0.44
```

### Visual Feature 5: Dataset Scatter Plot

For 2D datasets, display points by class.

Useful for:

```text
XOR
2D point classifier
classification demos
```

### Visual Feature 6: Decision Boundary

For 2D classifiers, show learned decision regions.

Implementation idea:

```text
sample many points on a 2D grid
run model.predict(point)
color the area based on predicted class
draw real dataset points over it
```

### Visualization Rule

Do not mix rendering logic with training logic.

The visualizer should read data from:

```text
TrainingHistory
Model summary
Tensor values
Dataset points
Predictions
```

---

## 20. Example Models

The project must include demo models.

### Demo 1: XOR Classifier

This is the required first serious demo.

Dataset:

```text
0 0 -> 0
0 1 -> 1
1 0 -> 1
1 1 -> 0
```

Suggested model:

```text
Input: 2
Linear: 2 → 4
Activation: Sigmoid or ReLU
Linear: 4 → 1
Activation: Sigmoid
Loss: MSE or BCE
Optimizer: SGD
```

Why XOR matters:

* It is simple.
* It is famous.
* It cannot be solved by a single linear layer.
* It proves hidden layers and nonlinear activation matter.

### Demo 2: Linear Regression

Use CSV data.

Example:

```text
hours_studied,previous_score,result
1,50,55
2,55,60
3,60,66
4,65,72
5,70,78
```

This demo proves:

```text
CSV loading
train/test split
regression
MSE loss
prediction
```

### Demo 3: 2D Point Classifier

Optional but very useful for visualization.

It proves:

```text
dataset scatter plot
classification
decision boundary visualization
```

---

## 21. Testing Requirements

Core math must be tested.

Required tests:

```text
Tensor creation
Shape validation
Matrix addition
Matrix multiplication
Transpose
ReLU output
Sigmoid output
MSE calculation
SGD parameter update
Linear forward shape
Sequential forward shape
XOR loss decreases during training
Save/load preserves parameters
```

Training tests should use deterministic random seeds.

Do not accept flaky tests.

The goal is not perfect accuracy every run. The goal is stable, reproducible evidence that training works.

---

## 22. Documentation Requirements

Maintain documentation in `docs/`.

Recommended docs:

```text
docs/architecture.md
docs/tensor.md
docs/autograd.md
docs/training.md
docs/visualization.md
docs/examples.md
docs/roadmap.md
```

The README should explain:

```text
what NeuroForge is
what it can do
what it cannot do
how to build
how to run tests
how to run XOR demo
how to run visualizer
```

---

## 23. Roadmap

Follow this order unless there is a strong reason not to.

### Phase 1: Project Skeleton

```text
CMake
folder structure
README
basic tests
namespace neuroforge
```

### Phase 2: Math Core

```text
Tensor or Matrix
shape validation
addition
subtraction
element-wise multiply
matmul
transpose
random initialization
tests
```

### Phase 3: Basic Neural Network

```text
Parameter
Module
Linear
ReLU
Sigmoid
Sequential
MSELoss
SGD
manual backward if needed
```

### Phase 4: First Working Training Demo

```text
XOR dataset
training loop
loss decreases
predictions printed
tests or reproducible example
```

### Phase 5: Autograd

```text
operation graph
backward engine
gradient accumulation
supported operations
autograd tests
```

### Phase 6: Data System

```text
Dataset
CSVDataset
DataLoader
train/test split
regression demo
```

### Phase 7: Serialization

```text
save model
load model
checkpoint
save/load tests
```

### Phase 8: Visualization

```text
Dear ImGui window
model architecture view
loss plot with ImPlot
gradient monitor
tensor inspector
dataset scatter plot
decision boundary if possible
```

### Phase 9: Polish

```text
clean API
docs
examples
better error messages
final demo
presentation material
```

---

## 24. Minimum Viable Product

The MVP is complete when NeuroForge can:

```text
create tensors/matrices
perform matrix multiplication
define a Sequential model
use Linear and activation layers
calculate MSE loss
train with SGD
solve or mostly solve XOR
show loss decreasing
save training history
run tests successfully
```

The visual MVP is complete when it can:

```text
open a GUI window
show model architecture
show training loss curve
show basic tensor or gradient info
```

---

## 25. Stretch Goals

Only work on these after the MVP works:

```text
Adam optimizer
BinaryCrossEntropyLoss
mini-batch DataLoader
CSV regression demo
decision boundary visualization
Graphviz export
model save/load
gradient checking
more activation functions
basic broadcasting
```

Do not start stretch goals before the core training loop works.

---

## 26. Forbidden Shortcuts

Do not:

```text
use PyTorch
use TensorFlow
use external autograd libraries
use external ML libraries
fake the training result
hardcode XOR predictions
skip tests for math operations
mix GUI code with core training code
make the architecture depend only on the demo
```

---

## 27. Error Handling Rules

Use clear exceptions for invalid operations.

Examples:

```text
Matrix multiplication shape mismatch
Invalid tensor index
CSV parse failure
Model layer shape mismatch
Optimizer received no parameters
Loss prediction/target shape mismatch
```

Good error message:

```text
MatMul shape mismatch: left shape [4, 2] cannot multiply right shape [3, 1].
```

Bad error message:

```text
Invalid operation.
```

---

## 28. Naming Rules

Use clear names.

Good names:

```text
Tensor
Linear
Sequential
MSELoss
SGD
Trainer
DataLoader
TrainingHistory
ModelSaver
```

Avoid vague names:

```text
Thing
Manager
Processor
Stuff
AIClass
Doer
```

---

## 29. API Design Rules

The user-facing API should be simple.

Prefer:

```cpp
Sequential model;
model.add(std::make_shared<Linear>(2, 4));
model.add(std::make_shared<ReLU>());
```

Avoid:

```cpp
NeuralNetLayerProcessingManager manager;
manager.registerLayerObjectWithInternalPointer(...);
```

The API should feel small, clean, and understandable.

---

## 30. Performance Rules

Correctness comes before performance.

Initial implementation may use simple loops.

Acceptable first implementation:

```text
O(n^3) matrix multiplication with clear loops
```

Do not use advanced SIMD, BLAS, CUDA, or complicated optimization early.

Later possible improvements:

```text
cache-friendly loops
parallelization
optional BLAS backend
```

Only optimize after tests pass.

---

## 31. Randomness Rules

Use deterministic seeds for tests and demos.

Example:

```cpp
Random::seed(42);
```

This makes training reproducible.

---

## 32. Demo Quality Bar

The final demo should clearly show:

```text
model architecture
dataset
training loss going down
final predictions
learned behavior
optional visual dashboard
```

For XOR, output should look like:

```text
Input: [0, 0] Prediction: 0.03 Target: 0
Input: [0, 1] Prediction: 0.96 Target: 1
Input: [1, 0] Prediction: 0.95 Target: 1
Input: [1, 1] Prediction: 0.04 Target: 0
```

Values do not need to be exact, but the behavior should be correct.

---

## 33. Task Ticket Format

When creating or dispatching a task, use this format:

```text
Task:
Context:
Goal:
Files likely involved:
Constraints:
Expected output:
Tests required:
Definition of done:
```

Example:

```text
Task:
Implement matrix multiplication.

Context:
NeuroForge needs reliable 2D matrix multiplication for Linear layers.

Goal:
Add Tensor::matmul with shape validation.

Files likely involved:
include/neuroforge/core/Tensor.hpp
src/core/Tensor.cpp
tests/test_tensor.cpp

Constraints:
No external math libraries.
Throw std::invalid_argument on incompatible shapes.

Expected output:
Working matmul implementation.

Tests required:
2x2 * 2x2
2x3 * 3x1
invalid shape throws
known numerical result

Definition of done:
All tensor tests pass and the implementation is readable.
```

---

## 34. Suggested Subagents

Use specialized subagents conceptually when the task is large.

The main agent remains responsible for integration.

Recommended subagents:

```text
Math Core Agent
Autograd Agent
Neural Network Agent
Training Agent
Data Agent
Visualization Agent
Testing Agent
Documentation Agent
Code Review Agent
```

Subagents should not make final architecture decisions alone.

---

## 35. Math Core Agent

Purpose:

Build and verify the tensor/matrix foundation.

Responsibilities:

```text
Tensor storage
Matrix operations
Shape checks
Indexing
Matmul
Transpose
Random initialization
Numerical stability
Core tests
```

Rules:

* Prioritize correctness over speed.
* Check dimension compatibility.
* Throw clear errors for invalid shapes.
* Write tests for every operation.
* Keep APIs simple.

---

## 36. Autograd Agent

Purpose:

Design and implement automatic differentiation.

Responsibilities:

```text
Computation graph
Operation nodes
Backward functions
Gradient accumulation
Topological backward order
Autograd tests
```

Rules:

* Start simple.
* Validate gradients with known derivatives.
* Clearly document which operations support autograd.
* Make `loss.backward()` understandable.
* Do not hide broken autograd behind manual hacks.

---

## 37. Neural Network Agent

Purpose:

Build the model-building API.

Responsibilities:

```text
Parameter
Module
Linear
ReLU
Sigmoid
Tanh
Sequential
model.parameters()
model.summary()
```

Rules:

* Keep API PyTorch-inspired but C++-friendly.
* Ensure layers expose parameters.
* Ensure forward pass shape is correct.
* Cache values needed for backward if manual backprop is used.

---

## 38. Training Agent

Purpose:

Implement the training loop.

Responsibilities:

```text
Trainer
TrainingConfig
TrainingHistory
loss tracking
metric tracking
epoch loop
batch loop
```

Rules:

* Make training simple to use.
* Store loss per epoch.
* Ensure optimizer zeroes gradients.
* Log enough information for debugging.
* Support the visualizer by exposing training history.

---

## 39. Data Agent

Purpose:

Implement datasets and dataloaders.

Responsibilities:

```text
Dataset
CSVDataset
Batch
DataLoader
TrainTestSplitter
Preprocessor
```

Rules:

* Start with numeric CSV only.
* Keep parsing strict and clear.
* Provide helpful error messages.
* Add tests with small sample CSV files.
* Do not overbuild preprocessing.

---

## 40. Visualization Agent

Purpose:

Build NeuroForge Visual Lab.

Responsibilities:

```text
Dear ImGui app
ImPlot loss graph
model architecture view
tensor inspector
gradient monitor
dataset scatter plot
decision boundary view
```

Rules:

* Keep visualization separate from core ML.
* The framework must run without GUI.
* Read data from clean interfaces such as `TrainingHistory`.
* Do not mix rendering logic with training logic.
* Make the demo visually impressive.

---

## 41. Testing Agent

Purpose:

Protect correctness.

Responsibilities:

```text
unit tests
integration tests
training behavior tests
regression tests
edge cases
```

Rules:

* Test math heavily.
* Test shape errors.
* Test gradient calculations.
* Test that XOR loss decreases.
* Use deterministic random seeds.
* Do not accept flaky tests.

---

## 42. Documentation Agent

Purpose:

Make the project understandable.

Responsibilities:

```text
README
architecture docs
API examples
demo instructions
comments where needed
diagrams
```

Rules:

* Explain the project simply.
* Explain why each module exists.
* Include build and run commands.
* Include example code.
* Keep docs updated with actual behavior.

---

## 43. Code Review Agent

Purpose:

Review code before integration.

Responsibilities:

```text
find bugs
find design problems
find unclear names
find missing tests
find memory issues
find overengineering
```

Rules:

* Be critical.
* Do not praise weak code.
* Check ownership and lifetimes.
* Check shape safety.
* Check whether the code fits the architecture.
* Suggest concrete fixes.

---

## 44. README Pitch

The README should describe NeuroForge like this:

```text
NeuroForge is a small educational C++ machine learning framework inspired by PyTorch. It implements tensors, matrix operations, neural network layers, loss functions, optimizers, training loops, and visualization tools from scratch. The goal is to demonstrate how machine learning frameworks work internally.
```

---

## 45. Final Success Criteria

NeuroForge is successful if:

```text
It builds cleanly.
It has tests.
It can train at least one real small model.
It has clear architecture.
It has many meaningful classes.
It demonstrates tensors, backpropagation, and optimization.
It has documentation.
It has a visual component or visual export.
It can be explained confidently.
```

The project should prove strong understanding of C++ OOP, linear algebra, and machine learning foundations.

---

## 46. Final Instruction

Always act like the technical CEO of NeuroForge.

Do not only complete the immediate coding request.

Also protect the project architecture.

Build NeuroForge step by step into a small but impressive C++ ML framework.
