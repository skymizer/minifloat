#include "minifloat.hpp"
#include <gtest/gtest.h>

using skymizer::Minifloat;
using N = skymizer::minifloat::NaNStyle;

TEST(SanityCheck, truncate) {
  EXPECT_EQ((Minifloat<3, 4>{2.0f}.bits()), 0x40);
  EXPECT_EQ((Minifloat<4, 3>{2.0f}.bits()), 0x40);
  EXPECT_EQ((Minifloat<5, 2>{2.0f}.bits()), 0x40);

  EXPECT_EQ((Minifloat<3, 4>{1.0f}.bits()), 0b0'011'0000);
  EXPECT_EQ((Minifloat<4, 3>{1.0f}.bits()), 0b0'0111'000);
  EXPECT_EQ((Minifloat<5, 2>{1.0f}.bits()), 0b0'01111'00);

  EXPECT_EQ((Minifloat<3, 4>{-1.25f}.bits()), 0b1'011'0100);
  EXPECT_EQ((Minifloat<4, 3>{-1.25f}.bits()), 0b1'0111'010);
  EXPECT_EQ((Minifloat<5, 2>{-1.25f}.bits()), 0b1'01111'01);
}

TEST(SanityCheck, equality) {
  EXPECT_EQ(float(Minifloat<3, 4>{-3.0f}), -3.0f);
  EXPECT_EQ(float(Minifloat<4, 3>{-3.0f}), -3.0f);
  EXPECT_EQ(float(Minifloat<5, 2>{-3.0f}), -3.0f);

  EXPECT_EQ(double(Minifloat<3, 4>{-3.0}), -3.0);
  EXPECT_EQ(double(Minifloat<4, 3>{-3.0}), -3.0);
  EXPECT_EQ(double(Minifloat<5, 2>{-3.0}), -3.0);

  Minifloat<3, 4> x(-3.0f);
  static_assert(std::is_same<decltype(x + x), float>::value);
  EXPECT_TRUE(x == x);
  EXPECT_FALSE(x > x);

  EXPECT_TRUE((Minifloat<4, 3, N::IEEE>{NAN}.is_nan()));
}