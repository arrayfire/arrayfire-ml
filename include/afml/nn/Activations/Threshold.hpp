/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <afml/nn/Activations/Activation.hpp>

namespace afml
{
    namespace nn
    {
        class ThresholdNode : public ActivationNode
        {
        private:
            double mVal;

            af::array fn(const af::array &input)
            {
                af::array cond = (input >= mVal);
                return (cond) * input + (1 - cond) * mVal;
            }

            af::array dfn(const af::array &input)
            {
                return (input >= mVal).as(input.type());
            }
        public:
            ThresholdNode(int size, double val, const char *name="none") :
                mVal(val),
                ActivationNode(size, name)
            {
            }
        };

        typedef ThresholdNode Threshold;
    }
}
