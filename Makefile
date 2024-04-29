CXXFLAGS = -std=c++17 -Wall -Wextra -pipe -march=native

check: test
	./test

test: test.cpp minifloat.hpp
	$(CXX) $(CXXFLAGS) -lgtest -lgtest_main -o $@ test.cpp