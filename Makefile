CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -march=native -O3
TEST_SOURCES := $(wildcard tests/*.cpp)

check: test
	if command -v gtest-parallel; then gtest-parallel ./test; else ./test; fi

test: $(TEST_SOURCES) tests/support.hpp minifloat.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -pthread -I. -o $@ $(TEST_SOURCES) -lgtest -lgtest_main

bench: benches/arith.cpp minifloat.hpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DNDEBUG -I. -o $@ $<

run-bench: bench
	if command -v taskset; then taskset -c 2 ./bench; else ./bench; fi

format:
	clang-format -i minifloat.hpp benches/*.cpp tests/*.cpp tests/*.hpp
