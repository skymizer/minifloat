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

template <typename T>
static void test_conversion() {
  using detail::bit_cast;
  const bool IS_FNUZ = Trait<T>::N == NaNStyle::FNUZ;

  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{0.0f})), 0);
  EXPECT_EQ(bit_cast<std::uint32_t>(id<float>(T{-0.0f})), !IS_FNUZ * 0x8000'0000);
  EXPECT_EQ(bit_cast<std::uint64_t>(id<double>(T{-0.0f})), !IS_FNUZ * 0x8000'0000'0000'0000);

  foreach<T>([](T x){
    if (x.is_nan()) return;
    EXPECT_EQ(x, T::from_bits(x.bits()));
    EXPECT_EQ(x, T{id<float>(x)});
    EXPECT_EQ(id<float>(x), id<double>(x));
  });
}

TEST(SanityCheck, conversion) {
  test_conversion<E3M4>();
  test_conversion<E3M4FN>();
  test_conversion<E3M4FNUZ>();

  test_conversion<E4M3>();
  test_conversion<E4M3FN>();
  test_conversion<E4M3FNUZ>();

  test_conversion<E5M2>();
  test_conversion<E5M2FN>();
  test_conversion<E5M2FNUZ>();

  test_conversion<E5M7>();
  test_conversion<E5M7FN>();
  test_conversion<E5M7FNUZ>();
}