cmake_minimum_required(VERSION 3.22)

project(qianli_infer_backend_27 LANGUAGES CXX)

find_package(fmt REQUIRED)
add_subdirectory(backend)

add_executable(test_fps src/test_fps.cpp)

target_link_libraries(test_fps PRIVATE backend fmt)
