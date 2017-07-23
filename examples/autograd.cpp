/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd.h>

#define VERIFY(VAL) do {                                    \
        auto res = af::allTrue<bool>(af::abs(VAL) < 1E-5);  \
        printf("%s:%d %s\n", __FUNCTION__, __LINE__,        \
               res ? "PASS" : "FAIL");                      \
    } while(0)

using af::autograd::Variable;
void test_multiply()
{
    auto x = Variable(af::randu(5), true);
    auto y = x * x;
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    VERIFY(dx.array() - 2 * x.array());
}

void test_multipl_add()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    VERIFY(dx.array() - 2 * x.array() - y.array());
    VERIFY(dy.array() - 2 * y.array() - x.array());
}

void test_no_calc_grad()
{
    auto x = Variable(af::randu(5), false);
    auto y = Variable(af::randu(5), true);
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    VERIFY(dy.array() - 2 * y.array() - x.array());
    try {
        auto dx = x.grad();
    } catch(af::exception &ex) {
        std::cout << ex.what() << std::endl;
        return;
    }
    printf("%s:%d No Gradient check Failed\n");
}

void test_multiply_sub()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x * x - x * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    VERIFY(dx.array() - (2 * x.array() - y.array()));
    VERIFY(dy.array() - (-x.array()));
}

void test_divide_add()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = x + x / y + y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    VERIFY(dx.array() - (1.0 + 1.0 / y.array()));
    VERIFY(dy.array() - (1.0 - x.array() / (y.array() * y.array())));
}

void test_multiply_add_scalar()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5), true);
    auto z = 2 * x + x * y + y;
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dx = x.grad();
    auto dy = y.grad();
    VERIFY(dx.array() - (2.0 + y.array()));
    VERIFY(dy.array() - (1.0 + x.array()));
}

void test_exp()
{
    auto x = Variable(af::randu(5), true);
    auto y = exp(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    VERIFY(dx.array() - (af::exp(x.array())));
}

void test_sigmoid()
{
    auto x = Variable(af::randu(5), true);
    auto y = sigmoid(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    VERIFY(dx.array() - (y.array() * (1 - y.array())));
    VERIFY(dx.array() - (af::sigmoid(x.array()) * (1 - af::sigmoid(x.array()))));
}

void test_tanh()
{
    auto x = Variable(af::randu(5), true);
    auto y = tanh(x);
    auto dy = Variable(af::constant(1.0, 5), false);
    y.backward(dy);
    auto dx = x.grad();
    VERIFY(dx.array() - (1 - y.array() * y.array()));
    VERIFY(dx.array() - (1 + af::tanh(x.array())) * (1 - af::tanh(x.array())));
}

void test_tile()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5, 2), true);
    auto z = y * tileAs(x, y);
    auto dz = Variable(af::constant(1.0, 5, 2), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    VERIFY(dy.array() - af::tile(x.array(), 1, 2));
    VERIFY(dx.array() - af::sum(y.array(), 1));
}

void test_sum()
{
    auto x = Variable(af::randu(5), true);
    auto y = Variable(af::randu(5, 2), true);
    auto z = x * sumAs(y, x);
    auto dz = Variable(af::constant(1.0, 5), false);
    z.backward(dz);
    auto dy = y.grad();
    auto dx = x.grad();
    VERIFY(dy.array() - af::tile(x.array(), 1, 2));
    VERIFY(dx.array() - af::sum(y.array(), 1));
}

int main()
{
    af::info();
    test_multiply();
    test_multipl_add();
    test_no_calc_grad();
    test_multiply_sub();
    test_divide_add();
    test_multiply_add_scalar();
    test_exp();
    test_sigmoid();
    test_tanh();
    test_tile();
    test_sum();
    return 0;
}
