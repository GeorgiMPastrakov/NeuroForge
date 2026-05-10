add_executable(neuroforge_smoke test_smoke.cpp)

target_link_libraries(neuroforge_smoke PRIVATE neuroforge)

add_test(NAME neuroforge_smoke COMMAND neuroforge_smoke)

add_executable(neuroforge_shape test_shape.cpp)

target_link_libraries(neuroforge_shape PRIVATE neuroforge)

add_test(NAME neuroforge_shape COMMAND neuroforge_shape)

add_executable(neuroforge_tensor test_tensor.cpp)

target_link_libraries(neuroforge_tensor PRIVATE neuroforge)

add_test(NAME neuroforge_tensor COMMAND neuroforge_tensor)

add_executable(neuroforge_nn test_nn.cpp)

target_link_libraries(neuroforge_nn PRIVATE neuroforge)

add_test(NAME neuroforge_nn COMMAND neuroforge_nn)

add_executable(neuroforge_losses test_losses.cpp)

target_link_libraries(neuroforge_losses PRIVATE neuroforge)

add_test(NAME neuroforge_losses COMMAND neuroforge_losses)

add_executable(neuroforge_optim test_optim.cpp)

target_link_libraries(neuroforge_optim PRIVATE neuroforge)

add_test(NAME neuroforge_optim COMMAND neuroforge_optim)

add_executable(neuroforge_training test_training.cpp)

target_link_libraries(neuroforge_training PRIVATE neuroforge)

add_test(NAME neuroforge_training COMMAND neuroforge_training)

add_executable(neuroforge_autograd test_autograd.cpp)

target_link_libraries(neuroforge_autograd PRIVATE neuroforge)

add_test(NAME neuroforge_autograd COMMAND neuroforge_autograd)

add_executable(neuroforge_data test_data.cpp)

target_link_libraries(neuroforge_data PRIVATE neuroforge)

add_test(NAME neuroforge_data COMMAND neuroforge_data)

add_executable(neuroforge_metrics test_metrics.cpp)

target_link_libraries(neuroforge_metrics PRIVATE neuroforge)

add_test(NAME neuroforge_metrics COMMAND neuroforge_metrics)

add_executable(neuroforge_serialization test_serialization.cpp)

target_link_libraries(neuroforge_serialization PRIVATE neuroforge)

add_test(NAME neuroforge_serialization COMMAND neuroforge_serialization)

add_executable(neuroforge_visualization test_visualization.cpp)

target_link_libraries(neuroforge_visualization PRIVATE neuroforge)

add_test(NAME neuroforge_visualization COMMAND neuroforge_visualization)
