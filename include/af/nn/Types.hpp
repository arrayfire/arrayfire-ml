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

namespace af {
    namespace nn {

        autograd::Variable input(const af::array &arr);

        autograd::Variable parameter(const af::array &arr);

        autograd::Variable weight(int input_size, int output_size, float spread = 0.05);
    }
}
