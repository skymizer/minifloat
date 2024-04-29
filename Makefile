CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pipe -march=native

check: test
	./test

test: test.cpp
	$(CXX) $(CXXFLAGS) -lgtest -lgtest_main -o $@ test.cpp