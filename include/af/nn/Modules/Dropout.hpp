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
        class Dropout : public Module
        {
        private:
            double m_ratio;
        public:
            Dropout(double drop_ratio = 0.5);

            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
