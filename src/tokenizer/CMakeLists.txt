cmake_minimum_required(VERSION 3.14)
project(aml_tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# fetch and configure third-party dependencies
include(FetchContent)

# fetch GoogleTest
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# fetch nlohmann/json
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.zip
)

# make both available
FetchContent_MakeAvailable(googletest nlohmann_json)

# build the Tokenizer as a static library
add_library(tokenizer aml_tokenizer.cpp)
target_link_libraries(tokenizer 
  PUBLIC 
  nlohmann_json::nlohmann_json
)

# setup testing executable
enable_testing()

add_executable(tokenizer_test test_tokenizer.cpp)
target_link_libraries(tokenizer_test
  PRIVATE
  tokenizer
  GTest::gtest_main
)

# register tests with CTest
include(GoogleTest)
gtest_discover_tests(tokenizer_test)

# fetch pybind11
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.zip
)
FetchContent_MakeAvailable(pybind11)

# create the Python module
pybind11_add_module(aml_tokenizer bindings.cpp)
target_link_libraries(aml_tokenizer PRIVATE tokenizer)