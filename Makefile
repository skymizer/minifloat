CXXFLAGS = -std=c++17 -Wall -Wextra -Wpedantic -march=native

check: test
	if command -v gtest-parallel; then gtest-parallel ./test; else ./test; fi

test: test.cpp minifloat.hpp
	-clang-format -i $^
	$(CXX) $(CXXFLAGS) -o $@ $< -lgtest -lgtest_main