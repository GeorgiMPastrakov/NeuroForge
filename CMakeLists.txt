cmake_minimum_required(VERSION 3.20)

project(NeuroForge VERSION 0.1.0 LANGUAGES CXX)

add_library(neuroforge INTERFACE)

target_compile_features(neuroforge INTERFACE cxx_std_20)

target_include_directories(neuroforge
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

enable_testing()

add_subdirectory(tests)
