/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/autograd/Variable.hpp>
#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Sigmoid : public Module
        {
        public:
            Sigmoid();

            autograd::Variable forward(const autograd::Variable &input);
        };

        class Tanh : public Module
        {
        public:
            Tanh();

            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
