.PHONY: build test examples visual run-visual validate clean

build:
	cmake -S . -B build
	cmake --build build

test: build
	ctest --test-dir build

examples: build
	./build/examples/neuroforge_xor
	./build/examples/neuroforge_linear_regression
	./build/examples/neuroforge_dense_classification
	./build/examples/neuroforge_population_regression

visual:
	cmake -S . -B build-visual -DNEUROFORGE_BUILD_VISUALIZER=ON
	cmake --build build-visual

run-visual: visual
	./build-visual/tools/visual_lab/neuroforge_visual_lab

validate: test examples visual
	rg "//|/\*|\*/" include src tests examples tools CMakeLists.txt || true
	git status -sb

clean:
	rm -rf build build-visual imgui.ini
