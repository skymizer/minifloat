#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer;

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

  EXPECT_TRUE((Minifloat<4, 3, minifloat::DefaultBias<4>::value, minifloat::NaNStyle::IEEE>{NAN}.is_nan()));
}