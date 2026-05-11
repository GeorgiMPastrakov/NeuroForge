# Validation

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

Check for C++ and CMake comments:

```bash
rg "//|/\*|\*/" include src tests examples tools CMakeLists.txt || true
```

Expected release state:

- all tests pass
- examples run
- regression demo saves `build/neuroforge_linear_regression_model.txt`
- optional visualizer builds
- no build artifacts are committed
- no duplicate `* 2.*` files are present in tracked source folders
