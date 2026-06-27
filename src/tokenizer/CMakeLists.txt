cmake_minimum_required(VERSION 3.14)
project(aml_tokenizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# build the Tokenizer as a static library
add_library(tokenizer aml_tokenizer.cpp)

# fetch and configure GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(tokenizer_test test_tokenizer.cpp)
target_link_libraries(tokenizer_test
  PRIVATE
  tokenizer
  GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tokenizer_test)