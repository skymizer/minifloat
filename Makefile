CXXFLAGS = -std=c++17 -Wall -Wextra -O3 -pipe -march=native

check: sanity.cpp
	$(CXX) $(CXXFLAGS) -lgtest -lgtest_main sanity.cpp