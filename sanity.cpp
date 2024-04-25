#include "minifloat.hpp"

int main() {
    [[maybe_unused]] const double x = skymizer::Minifloat<5, 10>(-0.7f);
    [[maybe_unused]] const double y(skymizer::Minifloat<15, 0>(3.14));
}