#include <gtest/gtest.h>

#include "tensor.hpp"


TEST(LinearTest, Determinant2x2)
{
  arch::tensor<float> t({2, 2}, {4, 6, 3, 8});
  auto                result = t.det();

  EXPECT_NEAR(result[0], 4 * 8 - 6 * 3, 1e-5);
}