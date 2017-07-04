/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd.h>

using af::autograd::Variable;
using af::autograd::backward;
void test1()
{
    auto x = Variable(af::randu(5), true);
    af_print(x.array());
    auto y = x * x;
    af_print(y.array());
    auto dy = Variable(af::constant(1.0, 5), false);
    backward(y, dy);
    auto dx = x.grad();
    af_print(dx.array() - 2 * x.array());
}

void test2()
{
    auto x = Variable(af::randu(5), true);
    af_print(x.array());
    auto y = Variable(af::randu(5), true);
    af_print(y.array());
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    backward(z, dz);
    auto dx = x.grad();
    auto dy = y.grad();
    af_print(dx.array() - 2 * x.array() - y.array());
    af_print(dy.array() - 2 * y.array() - x.array());
}

void test3()
{
    auto x = Variable(af::randu(5), false);
    af_print(x.array());
    auto y = Variable(af::randu(5), true);
    af_print(y.array());
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5), false);
    backward(z, dz);
    auto dy = y.grad();
    af_print(dy.array() - 2 * y.array() - x.array());
    try {
        auto dx = x.grad();
    } catch(af::exception &ex) {
        std::cout << ex.what() << std::endl;
    }
}

int main()
{
    af::info();
    test1();
    test2();
    test3();
    return 0;
}
