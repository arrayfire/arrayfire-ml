/*******************************************************
 * Copyright (c) 2018, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd.h>
#include <af/nn.h>
#include <gtest/gtest.h>
#include <iostream>

using af::autograd::Variable;

TEST(Autograd, Multiply) {
    auto x = Variable(af::randu(5), true);
    auto y = x * x;
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    EXPECT_TRUE(af::allTrue<bool>((dx.array() - 2 * x.array()) < 1E-5));
}
