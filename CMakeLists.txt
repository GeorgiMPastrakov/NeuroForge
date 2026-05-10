cmake_minimum_required(VERSION 3.20)

project(NeuroForge VERSION 0.1.0 LANGUAGES CXX)

add_library(neuroforge STATIC
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
