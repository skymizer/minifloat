#include "minifloat.hpp"
#include <gtest/gtest.h>

using namespace skymizer::minifloat;

template <typename T>
T id(T x) { return x; }

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

TEST(SanityCheck, equality) {
  EXPECT_EQ(id<float>(E3M4{-3.0f}), -3.0f);
  EXPECT_EQ(id<float>(E4M3{-3.0f}), -3.0f);
  EXPECT_EQ(id<float>(E5M2{-3.0f}), -3.0f);
  EXPECT_EQ(id<float>(E5M7{-3.0f}), -3.0f);

  EXPECT_EQ(id<double>(E3M4{-3.0}), -3.0);
  EXPECT_EQ(id<double>(E4M3{-3.0}), -3.0);
  EXPECT_EQ(id<double>(E5M2{-3.0}), -3.0);
  EXPECT_EQ(id<double>(E5M7{-3.0}), -3.0);

  EXPECT_TRUE(E4M3{NAN}.is_nan());
  EXPECT_TRUE(E4M3FN{NAN}.is_nan());
  EXPECT_TRUE(E4M3FNUZ{NAN}.is_nan());

  EXPECT_TRUE(E5M7{NAN}.is_nan());
  EXPECT_TRUE(E5M7FN{NAN}.is_nan());
  EXPECT_TRUE(E5M7FNUZ{NAN}.is_nan());
}