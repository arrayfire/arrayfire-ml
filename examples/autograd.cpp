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

int main()
{
    af::info();
    test_multiply();
    test_multipl_add();
    test_no_calc_grad();
    test_multiply_sub();
    test_divide_add();
    test_multiply_add_scalar();
    return 0;
}
