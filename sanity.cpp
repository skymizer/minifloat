#include "minifloat.hpp"

int main() {
    const double x = skymizer::Minifloat<5, 10>(-0.7f);
    const double y(skymizer::Minifloat<15, 0>(3.14));
}