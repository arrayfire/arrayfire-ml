/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd/Functions.hpp>
#include <af/nn/Modules/Activations.hpp>

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Sigmoid::Sigmoid() {}

        Variable Sigmoid::forward(const Variable &input)
        {
            return sigmoid(input);
        }

        Tanh::Tanh() {}

        Variable Tanh::forward(const Variable &input)
        {
            return tanh(input);
        }
    }
}
