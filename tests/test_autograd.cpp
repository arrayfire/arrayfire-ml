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

using af::allTrue;
using af::autograd::Variable;

TEST(Autograd, Multiply)
{
    auto x = Variable(af::randu(5), true);
    auto y = x * x;
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    EXPECT_TRUE(allTrue<bool>((dx.array() - 2 * x.array()) < 1E-5));
}

TEST(Autograd, MultiplyAdd)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = dx.array() - 2 * x.array() - y.array();
    auto diffy = dy.array() - 2 * y.array() - x.array();

    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, noCalcGrad)
{
    auto x = Variable(af::randu(5), false);
    auto y = Variable(af::randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();

    auto diffy = (dy.array() - 2 * y.array() - x.array());
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
    try {
        auto dx = x.grad();
    } catch(af::exception &ex) {
        std::cout << ex.what() << std::endl;
        return;
    }
    printf("%s:%d No Gradient check Failed\n");
}

TEST(Autograd, MultiplySub)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x * x - x * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (2 * x.array() - y.array()));
    auto diffy = (dy.array() - (-x.array()));

    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, DivideAdd)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x + x / y + y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (1.0 + 1.0 / y.array()));
    auto diffy = (dy.array() - (1.0 - x.array() / (y.array() * y.array())));

    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, MultiplyAddScalar)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = 2 * x + x * y + y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    auto diffx = (dx.array() - (2.0 + y.array()));
    auto diffy = (dy.array() - (1.0 + x.array()));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, Exp)
{
    auto x = Variable(af::randu(5), true);
    auto y = exp(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (af::exp(x.array())));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
}

TEST(Autograd, Sigmoid)
{
    auto x = Variable(af::randu(5), true);
    auto y = sigmoid(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (y.array() * (1 - y.array())));
    auto diffy = (dx.array() - (af::sigmoid(x.array()) * (1 - af::sigmoid(x.array()))));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, Tanh)
{
    auto x = Variable(af::randu(5), true);
    auto y = tanh(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (1 - y.array() * y.array()));
    auto diffy = (dx.array() - (1 + af::tanh(x.array())) * (1 - af::tanh(x.array())));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, Tile)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5, 2), true);
    auto z = y * tileAs(x, y);
    auto dz = Variable(af::constant(1.0, 5, 2), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - af::tile(x.array(), 1, 2));
    auto diffy = (dx.array() - af::sum(y.array(), 1));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, Sum)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5, 2), true);
    auto z = x * sumAs(y, x);
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - af::tile(x.array(), 1, 2));
    auto diffy = (dx.array() - af::sum(y.array(), 1));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}

TEST(Autograd, Mean)
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5, 3, 2), true);
    auto z = x * mean(y, {1,2});
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    auto diffx = (dy.array() - 6 * af::tile(x.array(), 1, 3, 2));
    auto diffy = (dx.array() - af::mean(af::mean(y.array(), 1), 2));
    EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}
