/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/common.hpp>
#include <af/nn/Nodes/Node.hpp>

namespace af
{

    namespace nn
    {
        class ActivationNode : public Node
        {
        protected:

            virtual af::array fn(const af::array &val)
            {
                return val;
            }

            virtual af::array dfn(const af::array &val)
            {
                return af::constant(1, val.dims());
            }

        public:

            ActivationNode(int size, const char *name="none") :
                Node(1, &size, 1, &size, name)
            {
            }

            ArrayVector forward(const ArrayVector &input)
            {
                return { fn(input[0]) };
            }

            ArrayVector backward(const ArrayVector &input,
                                 const ArrayVector &gradOutput)
            {
                return { gradOutput[0] * dfn(input[0]) };
            }
        };

        typedef ActivationNode Activation;
    }
}
