/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Linear : public Module
        {
        private:
            bool m_bias;
        public:
            Linear(int input_size, int output_size, bool bias = true, float spread = 0.05);

            Linear(const autograd::Variable &w);

            Linear(const autograd::Variable &w, const autograd::Variable &b);

            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
