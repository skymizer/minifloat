CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -march=native -O3

check: test
	if command -v gtest-parallel; then gtest-parallel ./test; else ./test; fi

test: test.cpp minifloat.hpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lgtest -lgtest_main

format:
	clang-format -i *.cpp *.hpp