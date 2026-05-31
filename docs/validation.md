# Validation

Run the complete automated validation:

```bash
make validate
```

Remove generated build output:

```bash
make clean
```

Run the default build and tests:

```bash
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

Run examples:

```bash
./build/examples/neuroforge_xor
./build/examples/neuroforge_linear_regression
./build/examples/neuroforge_dense_classification
./build/examples/neuroforge_population_regression
```

Build the optional visualizer:

```bash
cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
cmake --build build-visual
```

Run the optional visualizer:

```bash
./build-visual/tools/visual_lab/neuroforge_visual_lab
```

Manual Visual Lab check:

- header shows `XOR Classification Demo`
- graph shows `2 -> 4 -> 1` neurons and weighted connections
- loss curve converges
- decision map overlays four XOR truth-table samples
- prediction table classifies `0, 1, 1, 0`
- dashboard fits without scrolling or ImGui warnings

Check for C++ and CMake comments:

```bash
rg "//|/\*|\*/" include src tests examples tools CMakeLists.txt || true
```

Expected release state:

- all tests pass
- examples run
- dense classification reaches perfect accuracy on its deterministic demo dataset
- population regression estimates approximately `154400` people for 2025
- regression demo saves `build/neuroforge_linear_regression_model.txt`
- optional visualizer builds
- Visual Lab opens as a display-only XOR dashboard
- Visual Lab shows the network graph, loss curve, decision map, and prediction table without scrolling
- no build artifacts are committed
- no duplicate `* 2.*` files are present in tracked source folders
