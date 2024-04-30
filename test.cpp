#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer::minifloat;

namespace {
template <typename> struct Trait;

template <int E_, int M_, NaNStyle N_, int B_, SubnormalStyle D_>
struct Trait<Minifloat<E_, M_, N_, B_, D_>> {
  static const int E = E_;
  static const int M = M_;
  static const NaNStyle N = N_;
  static const int B = B_;
  static const SubnormalStyle D = D_;
};
} // namespace

template <typename T, typename F>
static void foreach(F f) {
  const int WIDTH = Trait<T>::E + Trait<T>::M + 1;

  for (unsigned i = 0; i < (1U << WIDTH); ++i)
    f(T::from_bits(i));
}

template <typename T>
static T id(T x) { return x; }

static const struct {
  bool operator()(double x, double y) const {
    using detail::bit_cast;
    return bit_cast<std::uint64_t>(x) == bit_cast<std::uint64_t>(y) || (x != x && y != y);
  }

  bool operator()(float x, float y) const {
    using detail::bit_cast;
    return bit_cast<std::uint32_t>(x) == bit_cast<std::uint32_t>(y) || (x != x && y != y);
  }

  template <int E, int M, NaNStyle N, int B, SubnormalStyle D>
  bool operator()(Minifloat<E, M, N, B, D> x, Minifloat<E, M, N, B, D> y) const {
    return x.bits() == y.bits() || (x.is_nan() && y.is_nan());
  }
} are_identical;

#define RUN_ON_SELECTED_TYPES(F) do { \
  F<E3M4>(); \
  F<E3M4FN>(); \
  F<E3M4FNUZ>(); \
  F<E4M3>(); \
  F<E4M3FN>(); \
  F<E4M3FNUZ>(); \
  F<E5M2>(); \
  F<E5M2FN>(); \
  F<E5M2FNUZ>(); \
  F<E5M7>(); \
  F<E5M7FN>(); \
  F<E5M7FNUZ>(); \
} while (false)

template <int E, int M>
static void test_finite_bits(float x, unsigned bits) {
  EXPECT_EQ((Minifloat<E, M>{x}.bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NaNStyle::FN>{x}.bits()), bits);
  EXPECT_EQ((Minifloat<E, M, NaNStyle::FNUZ>{x}.bits()), bits);
}

TEST(SanityCheck, finite_bits) {
  test_finite_bits<3, 4>(2.0f, 0x40);
  test_finite_bits<4, 3>(2.0f, 0x40);
  test_finite_bits<5, 2>(2.0f, 0x40);
  test_finite_bits<5, 7>(2.0f, 0b0'10000'0000000);

  test_finite_bits<3, 4>(1.0f, 0b0'011'0000);
  test_finite_bits<4, 3>(1.0f, 0b0'0111'000);
  test_finite_bits<5, 2>(1.0f, 0b0'01111'00);
  test_finite_bits<5, 7>(1.0f, 0b0'01111'0000000);

  test_finite_bits<3, 4>(-1.25f, 0b1'011'0100);
  test_finite_bits<4, 3>(-1.25f, 0b1'0111'010);
  test_finite_bits<5, 2>(-1.25f, 0b1'01111'01);
  test_finite_bits<5, 7>(-1.25f, 0b1'01111'0100000);
}

template <typename T>
static void test_equality() {
  EXPECT_EQ(id<float>(T{-3.0f}), -3.0f);
  EXPECT_EQ(id<double>(T{-3.0}), -3.0);
  EXPECT_EQ(T{0.0f}, T{-0.0f});
  EXPECT_EQ(T{0.0f}.bits() == T{-0.0f}.bits(), Trait<T>::N == NaNStyle::FNUZ);
  EXPECT_TRUE(T{NAN}.is_nan());
  EXPECT_TRUE((std::isnan)(id<float>(T{NAN})));
  EXPECT_TRUE((std::isnan)(id<double>(T{NAN})));
  foreach<T>([](T x){ EXPECT_EQ(x != x, x.is_nan()); });
}

TEST(SanityCheck, equality) {
  RUN_ON_SELECTED_TYPES(test_equality);
}

template <typename T>
static void test_identity_conversion() {
  using detail::bit_cast;
  const bool IS_FNUZ = Trait<T>::N == NaNStyle::FNUZ;

  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{-0.0f})), !IS_FNUZ * 0x8000'0000);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{-0.0f})), !IS_FNUZ * 0x8000'0000'0000'0000);

  foreach<T>([](T x){
    EXPECT_PRED2(are_identical, x, T::from_bits(x.bits()));
    EXPECT_PRED2(are_identical, x, T{id<float>(x)});
    EXPECT_PRED2(are_identical, id<float>(x), id<double>(x));
  });
}

TEST(ConversionCheck, identity) {
  RUN_ON_SELECTED_TYPES(test_identity_conversion);
}