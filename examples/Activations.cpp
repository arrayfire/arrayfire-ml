/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/nn/Activations.hpp>

using namespace af::nn;

int main()
{
    const int num = 5;

    af::ArrayVector in = {100 * af::randu(num, 1) - 50};
    af::ArrayVector grad = {100 * af::randu(num, 1)};

    ReLU    r = ReLU(num, 0);
    Sigmoid s = Sigmoid(num);
    Tanh    t = Tanh(num);

    af_print(in[0]);
    af_print(r.forward(in)[0]);
    af_print(s.forward(in)[0]);
    af_print(t.forward(in)[0]);

    af_print(r.backward(in, grad)[0]);
    af_print(s.backward(in, grad)[0]);
    af_print(t.backward(in, grad)[0]);
}
