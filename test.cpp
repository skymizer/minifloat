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

TEST(SanityCheck, truncate) {
  EXPECT_EQ(E3M4{2.0f}.bits(), 0x40);
  EXPECT_EQ(E4M3{2.0f}.bits(), 0x40);
  EXPECT_EQ(E5M2{2.0f}.bits(), 0x40);
  EXPECT_EQ(E5M7{2.0f}.bits(), 0b0'10000'0000000);

  EXPECT_EQ(E3M4{1.0f}.bits(), 0b0'011'0000);
  EXPECT_EQ(E4M3{1.0f}.bits(), 0b0'0111'000);
  EXPECT_EQ(E5M2{1.0f}.bits(), 0b0'01111'00);
  EXPECT_EQ(E5M7{1.0f}.bits(), 0b0'01111'0000000);

  EXPECT_EQ(E3M4{-1.25f}.bits(), 0b1'011'0100);
  EXPECT_EQ(E4M3{-1.25f}.bits(), 0b1'0111'010);
  EXPECT_EQ(E5M2{-1.25f}.bits(), 0b1'01111'01);
  EXPECT_EQ(E5M7{-1.25f}.bits(), 0b1'01111'0100000);
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