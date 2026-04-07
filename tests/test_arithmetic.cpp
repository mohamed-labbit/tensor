#include <gtest/gtest.h>

#include "tensor.hpp"


TEST(TensorTest, EmptyTensorOps)
{
  arch::tensor<double> t;
  EXPECT_NO_THROW(t.sin());
  EXPECT_NO_THROW(t.tanh());
  EXPECT_EQ(t.sin().size(0), 0);
}