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
    auto x = Variable(af::randu(5));
    af_print(x.getData());
    auto y = x * x;
    af_print(y.getData());
    auto dy = Variable(af::constant(1.0, 5));
    backward(y, dy);
    af_print(x.getGrad().getData() - 2 * x.getData());
}

void test2()
{
    auto x = Variable(af::randu(5));
    af_print(x.getData());
    auto y = Variable(af::randu(5));
    af_print(y.getData());
    auto z = x * x + x * y + y * y;
    auto dz = Variable(af::constant(1.0, 5));
    backward(z, dz);
    af_print(x.getGrad().getData() - 2 * x.getData() - y.getData());
    af_print(y.getGrad().getData() - 2 * y.getData() - x.getData());
}

int main()
{
    af::info();
    test1();
    test2();
    return 0;
}
