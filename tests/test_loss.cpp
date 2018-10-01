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
using std::log;

using namespace af::nn;

TEST(Loss, MSE_1D)
{
    auto x = Variable(af::constant(2.0, 5), true);
    auto y = Variable(af::constant(0.0, 5), false);

    auto loss = MeanSquaredError();
    auto l = loss(x, y);

    ASSERT_TRUE(allTrue<bool>((l.array() - af::constant(4.0, 1)) < 1E-5));
    //TODO: Test gradient calculation...
}

TEST(Loss, MSE_nD)
{
    auto x = Variable(af::constant(1.0, 5, 100, 100, 100), true);
    auto y = Variable(af::constant(0.0, 5, 100, 100, 100), false);

    auto loss = MeanSquaredError();
    auto l = loss(x, y);

    ASSERT_TRUE(allTrue<bool>((l.array() - af::constant(1.0, 1)) < 1E-5));
    //TODO: Test gradient calculation...
}

TEST(Loss, MAE_1D)
{
    auto x = Variable(af::constant(2.0, 5), true);
    auto y = Variable(af::constant(0.0, 5), false);

    auto loss = MeanAbsoluteError();
    auto l = loss(x, y);

    ASSERT_TRUE(allTrue<bool>((l.array() - af::constant(2.0, 1)) < 1E-5));
    //TODO: Test gradient calculation...
}

TEST(Loss, MAE_nD)
{
    auto x = Variable(af::constant(2.0, 5, 100, 100, 100), true);
    auto y = Variable(af::constant(0.0, 5, 100, 100, 100), false);

    auto loss = MeanAbsoluteError();
    auto l = loss(x, y);

    ASSERT_TRUE(allTrue<bool>((l.array() - af::constant(2.0, 1)) < 1E-5));
    //TODO: Test gradient calculation...
}

TEST(Loss, BCELoss)
{
    auto x = Variable(af::constant(0.5, 5), true);
    auto y = Variable(af::constant(1.0, 5), false);

    auto loss = BinaryCrossEntropyLoss();
    auto l = loss(x, y);

    ASSERT_TRUE(allTrue<bool>((l.array() - af::constant(-std::log(0.5) , 1)) < 1E-5));
    //TODO: Test gradient calculation...
}

TEST(Loss, CELoss)
{
    auto x = Variable(af::transpose(af::range(5) + 0.5), true); //scores for each of C classes
    auto y = Variable(af::constant(2, 1), false); //vector of correct class labels

    auto loss = CELoss();
    auto l = loss(x, y);
    l.backward();

    af_print(l.array())
    af_print(x.grad().array())

    float h_x[] = { 0.0117,  0.0317, -0.9139,  0.2341,  0.6364 };
    af::array expected_grad(1, 5, h_x);

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-4));
    EXPECT_TRUE(allTrue<bool>(abs(l.array() - af::constant(2.4519, 1)) < 1E-4));
}

TEST(Loss, MultiMarginLoss)
{
    auto x = Variable(af::transpose(af::range(5) + 0.5), true); //scores for each of C classes
    auto y = Variable(af::constant(2, 1), false); //vector of correct class labels

    auto loss = MultiMarginLoss();
    auto l = loss(x, y);
    l.backward();

    float h_x[] = { 0.0, 0.0, -0.4, 0.2, 0.2 };
    af::array expected_grad(1, 5, h_x);

    auto diff = (x.grad().array() - expected_grad);
    EXPECT_TRUE(allTrue<bool>(abs(diff) < 1E-5));
    EXPECT_TRUE(allTrue<bool>(abs(l.array() - af::constant(1.0, 1)) < 1E-5));
}
