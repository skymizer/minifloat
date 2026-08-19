#include <minifloat.hpp>

#include <limits>

#if defined(SKYMIZER_MINIFLOAT_CONST) || defined(SKYMIZER_MINIFLOAT_PURE)
#error "minifloat.hpp leaked private helper macros"
#endif

int main() {
  using skymizer::minifloat::E4M3;

  const E4M3 value{1.5F};
  return value.to_float() == 1.5F && std::numeric_limits<E4M3>::is_specialized ? 0 : 1;
}
