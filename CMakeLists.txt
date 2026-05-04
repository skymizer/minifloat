cmake_minimum_required(VERSION 3.14)
project(skymizer-minifloat
  VERSION 0.1.0
  DESCRIPTION "Header-only C++ template library for minifloats"
  LANGUAGES CXX)

add_library(skymizer-minifloat INTERFACE)
add_library(skymizer::minifloat ALIAS skymizer-minifloat)

target_include_directories(skymizer-minifloat INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:include>)

target_compile_features(skymizer-minifloat INTERFACE cxx_std_17)

option(SKYMIZER_MINIFLOAT_BUILD_TESTS
  "Build the Google Test suite for skymizer-minifloat" OFF)

if(SKYMIZER_MINIFLOAT_BUILD_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(skymizer-minifloat-test test.cpp)
  target_link_libraries(skymizer-minifloat-test
    PRIVATE skymizer::minifloat GTest::gtest GTest::gtest_main)
  add_test(NAME skymizer-minifloat-test COMMAND skymizer-minifloat-test)
endif()

include(GNUInstallDirs)
install(FILES minifloat.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS skymizer-minifloat
  EXPORT skymizer-minifloat-targets)
install(EXPORT skymizer-minifloat-targets
  FILE skymizer-minifloat-config.cmake
  NAMESPACE skymizer::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/skymizer-minifloat)
