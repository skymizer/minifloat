cmake_minimum_required(VERSION 3.14)
project(skymizer-minifloat
  # 0.2.0-dev until the release is tagged; CMake versions carry no suffix.
  VERSION 0.2.0
  DESCRIPTION "Header-only C++ template library for minifloats"
  LANGUAGES CXX)

include(GNUInstallDirs)

add_library(skymizer-minifloat INTERFACE)
add_library(skymizer::minifloat ALIAS skymizer-minifloat)
set_target_properties(skymizer-minifloat PROPERTIES
  EXPORT_NAME minifloat)

target_include_directories(skymizer-minifloat INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_compile_features(skymizer-minifloat INTERFACE cxx_std_17)

option(SKYMIZER_MINIFLOAT_BUILD_TESTS
  "Build the Google Test suite for skymizer-minifloat" OFF)
option(SKYMIZER_MINIFLOAT_FETCH_GTEST
  "Download GoogleTest when building the test suite" OFF)

if(SKYMIZER_MINIFLOAT_BUILD_TESTS)
  enable_testing()

  if(SKYMIZER_MINIFLOAT_FETCH_GTEST)
    include(FetchContent)
    if(POLICY CMP0135)
      cmake_policy(SET CMP0135 NEW)
    endif()
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
      URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
    FetchContent_MakeAvailable(googletest)
  else()
    find_package(GTest REQUIRED)
  endif()

  add_executable(skymizer-minifloat-test test.cpp)
  target_link_libraries(skymizer-minifloat-test
    PRIVATE skymizer::minifloat GTest::gtest_main)
  add_test(NAME skymizer-minifloat-test COMMAND skymizer-minifloat-test)
endif()

install(FILES minifloat.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE
  DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/${PROJECT_NAME})
install(TARGETS skymizer-minifloat
  EXPORT skymizer-minifloat-targets)
install(EXPORT skymizer-minifloat-targets
  FILE skymizer-minifloat-config.cmake
  NAMESPACE skymizer::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/skymizer-minifloat)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/skymizer-minifloat-config-version.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMinorVersion
  ARCH_INDEPENDENT)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/skymizer-minifloat-config-version.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/skymizer-minifloat)
