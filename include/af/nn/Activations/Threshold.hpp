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
        class ThresholdNode : public ActivationNode
        {
        private:
            float mVal;

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
            ThresholdNode(int size, float val, const char *name="none") :
                ActivationNode(size, name),
                mVal(val)
            {
            }
        };

        typedef ThresholdNode Threshold;
    }
}
