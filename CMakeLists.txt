cmake_minimum_required(VERSION 3.20)

project(NeuroForge VERSION 0.1.0 LANGUAGES CXX)

add_library(neuroforge STATIC
    src/autograd/Value.cpp
    src/losses/BinaryCrossEntropyLoss.cpp
    src/losses/CrossEntropyLoss.cpp
    src/losses/MAELoss.cpp
    src/losses/MSELoss.cpp
    src/nn/Linear.cpp
    src/nn/Module.cpp
    src/nn/Parameter.cpp
    src/nn/ReLU.cpp
    src/nn/Sequential.cpp
    src/nn/Sigmoid.cpp
    src/nn/Tanh.cpp
    src/optim/Adam.cpp
    src/optim/MomentumSGD.cpp
    src/optim/SGD.cpp
    src/serialization/ModelLoader.cpp
    src/serialization/ModelSaver.cpp
    src/training/Trainer.cpp
    src/training/TrainingHistory.cpp
    src/core/Random.cpp
    src/core/Shape.cpp
    src/core/Tensor.cpp
    src/data/CSVDataset.cpp
    src/data/DataLoader.cpp
    src/data/Dataset.cpp
    src/data/TrainTestSplitter.cpp
    src/metrics/Metrics.cpp
    src/visualization/VisualizationData.cpp
)

target_compile_features(neuroforge PUBLIC cxx_std_20)

target_include_directories(neuroforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

enable_testing()

add_subdirectory(tests)
add_subdirectory(examples)

option(NEUROFORGE_BUILD_VISUALIZER "Build NeuroForge Visual Lab" OFF)

if(NEUROFORGE_BUILD_VISUALIZER)
    add_subdirectory(tools/visual_lab)
endif()
