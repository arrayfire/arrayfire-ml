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
    auto diff = dx.array() - 2 * x.array();
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
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

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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

    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

TEST(Autograd, Exp)
{
    auto x = Variable(af::randu(5), true);
    auto y = exp(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    auto diffx = (dx.array() - (af::exp(x.array())));
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}

/*
TEST(Autograd, Softmax)
{
    auto x = Variable(af::randu(5), true);
    auto x1a = x.array();
    x1a(0) += 0.1;
    auto x1 = Variable(x1a, true);

    auto y = softmax(x);
    auto y1 = softmax(x1);

    //auto dy = Variable(af::constant(1.0, 5), false);
    //y.backward(dy);
    y.backward();
    y1.backward();
    auto dx = x.grad();

    af_print(x.array());
    af_print(y.array());
    af_print(x1.array());
    af_print(y1.array());
    printf("distribution sums to 1? %f\n", af::sum<double>(y.array()));
    af_print(dx.array());

    //auto diffx = (dx.array() - (y.array() * (1 - y.array())));
    //auto diffy = (dx.array() - (af::sigmoid(x.array()) * (1 - af::sigmoid(x.array()))));
    //EXPECT_TRUE(allTrue<bool>(diffx < 1E-5));
    //EXPECT_TRUE(allTrue<bool>(diffy < 1E-5));
}
*/

TEST(Autograd, set_index)
{
    auto x   = Variable(af::range(5) + 0.5, true);
    auto idx = Variable(af::range(2) + 1, false);

    auto y = set_index(x, idx, Variable(af::constant(-2.0, idx.dims()), false));
    auto z = sum(2*y, {0});
    z.backward();

    auto expected_grad = constant(2, x.dims());
    expected_grad(idx.array()) = 0;

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
}

TEST(Autograd, select_index)
{
    auto x   = Variable(af::randu(5), true);
    auto idx = Variable(af::range(2) + 1, false);

    auto y = select_index(x, idx);
    auto z = sum(2*y, {0});
    z.backward();

    auto expected_grad = constant(0, x.dims());
    expected_grad(idx.array()) = 2;

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
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
    EXPECT_TRUE(allTrue<bool>(abs(diffx) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(diffy) < 1E-5));
}