cmake_minimum_required(VERSION 3.16)
project(echoscribe VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra)

add_library(echoscribe_core
    src/tensor/tensor.cpp
    src/gguf/gguf.cpp
    src/gguf/loader.cpp
    src/frontend/fft.cpp
    src/frontend/mel.cpp
    src/frontend/logmel.cpp
    src/ops/matmul.cpp
    src/ops/elementwise.cpp
    src/ops/conv1d.cpp
    src/ops/attention.cpp
)
target_include_directories(echoscribe_core PUBLIC src)

add_executable(echoscribe
    src/cli/main.cpp
)
target_link_libraries(echoscribe PRIVATE echoscribe_core)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
