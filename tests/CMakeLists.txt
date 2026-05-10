add_executable(neuroforge_smoke test_smoke.cpp)

target_link_libraries(neuroforge_smoke PRIVATE neuroforge)

add_test(NAME neuroforge_smoke COMMAND neuroforge_smoke)
