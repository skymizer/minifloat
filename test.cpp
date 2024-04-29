#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer::minifloat;

namespace {
template <typename> struct Trait;

template <unsigned E_, unsigned M_, NaNStyle N_, int B_, SubnormalStyle D_>
struct Trait<Minifloat<E_, M_, N_, B_, D_>> {
  static const unsigned E = E_;
  static const unsigned M = M_;
  static const NaNStyle N = N_;
  static const int B = B_;
  static const SubnormalStyle D = D_;
};
} // namespace

template <typename T>
static T id(T x) { return x; }

template <unsigned E, unsigned M>
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
  EXPECT_TRUE(std::isnan(id<float>(T{NAN})));
  EXPECT_TRUE(std::isnan(id<double>(T{NAN})));
}

TEST(SanityCheck, equality) {
  test_equality<E3M4>();
  test_equality<E3M4FN>();
  test_equality<E3M4FNUZ>();

  test_equality<E4M3>();
  test_equality<E4M3FN>();
  test_equality<E4M3FNUZ>();

  test_equality<E5M2>();
  test_equality<E5M2FN>();
  test_equality<E5M2FNUZ>();

  test_equality<E5M7>();
  test_equality<E5M7FN>();
  test_equality<E5M7FNUZ>();
}