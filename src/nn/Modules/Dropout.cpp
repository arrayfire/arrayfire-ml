/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/autograd/Functions.hpp>

#include <af/nn/Init.hpp>
#include <af/nn/Modules/Dropout.hpp>

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Dropout::Dropout(dim4 shape, double drop_ratio)
        {
            mask = nn::lecunUniform(shape) > drop_ratio;
        }

        Dropout::Dropout(const Variable &m) :
            mask(m)
        {
        }

        Variable Dropout::forward(const Variable &input)
        {
            return mask * input;
        }
    }
}
