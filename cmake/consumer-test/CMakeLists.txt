cmake_minimum_required(VERSION 3.14)
project(skymizer-minifloat-consumer LANGUAGES CXX)

find_package(skymizer-minifloat 0.1.0 CONFIG REQUIRED)

add_executable(skymizer-minifloat-consumer main.cpp)
target_link_libraries(skymizer-minifloat-consumer
  PRIVATE skymizer::minifloat)

enable_testing()
add_test(NAME skymizer-minifloat-consumer
  COMMAND skymizer-minifloat-consumer)
