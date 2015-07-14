/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afml/nn/Nodes/Node.hpp>

using namespace afml::nn;

int main()
{
    int inSize = 10;
    int outSize = 2;

    Node n(1, &inSize, 1, &outSize, "test");
    n.info();
}
