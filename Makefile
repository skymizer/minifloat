CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -march=native -O3
TEST_SOURCES := $(wildcard tests/*.cpp)

check: test
	if command -v gtest-parallel; then gtest-parallel ./test; else ./test; fi

test: $(TEST_SOURCES) tests/support.hpp minifloat.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -I. -o $@ $(TEST_SOURCES) -lgtest -lgtest_main

format:
	clang-format -i minifloat.hpp tests/*.cpp tests/*.hpp
