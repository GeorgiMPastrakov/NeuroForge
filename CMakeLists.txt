cmake_minimum_required(VERSION 3.20)

project(NeuroForge VERSION 0.1.0 LANGUAGES CXX)

add_library(neuroforge STATIC
    src/losses/MSELoss.cpp
    src/nn/Linear.cpp
    src/nn/Module.cpp
    src/nn/Parameter.cpp
    src/nn/ReLU.cpp
    src/nn/Sequential.cpp
    src/nn/Sigmoid.cpp
    src/nn/Tanh.cpp
    src/optim/SGD.cpp
    src/training/Trainer.cpp
    src/training/TrainingHistory.cpp
    src/core/Random.cpp
    src/core/Shape.cpp
    src/core/Tensor.cpp
)

target_compile_features(neuroforge PUBLIC cxx_std_20)

target_include_directories(neuroforge
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

enable_testing()

add_subdirectory(tests)
