/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afml/nn/Weights.hpp>

using namespace afml::nn;

int main()
{
    Weights w(10, 1, 0.05);
    af_print(w);

    return 0;
}
