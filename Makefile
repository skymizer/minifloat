CXXFLAGS = -std=c++17 -Wall -Wextra -pipe -march=native

check: test
	./test

test: test.cpp | minifloat.hpp
	$(CXX) $(CXXFLAGS) -o $@ $^ -lgtest -lgtest_main