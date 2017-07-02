/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/Activations/Activation.hpp>

namespace af
{
    namespace nn
    {
        class SigmoidNode : public ActivationNode
        {
        private:

            af::array fn(const af::array &input)
            {
                // TODO: replace with af::sigmoid
                return 1 / (1 + af::exp(-input));
            }

            af::array dfn(const af::array &input)
            {
                af::array output = fn(input);
                return output * (1 - output);
            }

        public:

            SigmoidNode(int size, const char *name="none") :
                ActivationNode(size, name)
            {
            }
        };

        typedef SigmoidNode Sigmoid;
    }
}
