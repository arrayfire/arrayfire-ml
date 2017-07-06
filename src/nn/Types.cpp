/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cmath>

#include <af/nn/Types.hpp>

namespace af {
    namespace nn {

        using autograd::Variable;

        Variable input(const af::array &arr)
        {
            return Variable(arr, false);
        }

        Variable parameter(const af::array &arr)
        {
            return Variable(arr, true);
        }

        Variable weight(int input_size, int output_size, float spread)
        {
            auto w = af::randu(output_size, input_size) * spread - spread / 2;
            w.eval();
            return parameter(w);
        }
    }
}
