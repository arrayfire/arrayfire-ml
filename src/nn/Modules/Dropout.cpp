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

        Dropout::Dropout(double drop_ratio) :
            m_ratio(drop_ratio)
        {
        }

        Variable Dropout::forward(const Variable &input)
        {
            if(m_train)
                return (uniform(input.dims(), 0.0, 1.0, f32, false) > m_ratio) * input;
            else
                return input;
        }
    }
}
