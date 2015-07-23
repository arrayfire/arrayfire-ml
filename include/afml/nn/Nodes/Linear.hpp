/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <afml/util/common.hpp>
#include <afml/nn/Weights.hpp>
#include <afml/nn/Nodes/Node.hpp>

namespace afml
{
    namespace nn
    {
        class LinearNode : public Node
        {
        private:

            Weights mWeights;
            Weights mDiffs;

        public:

            LinearNode(const int inputSize, const int outputSize,
                       double spread = 0.05,
                       const char *name="none") :
                Node(1, &inputSize, 1, &outputSize, name),
                mWeights(inputSize, outputSize, spread),
                mDiffs()
            {
            }

            ArrayVector forward(const ArrayVector &input)
            {
                return {af::matmul(mWeights.getWeights(), input[0]) +
                        af::tile(mWeights.getBias(), 1, input[0].dims(1))};
            }

            ArrayVector backward(const ArrayVector &input,
                                 const ArrayVector &gradOutput)
            {
                double m = input[0].dims(1);
                mDiffs.setWeights(af::matmulNT(gradOutput[0], input[0]) / m);
                mDiffs.setBias(af::sum(gradOutput[0], 1) / m);

                return { af::matmulTN(mWeights.getWeights(), gradOutput[0]) };
            }

            void update(double lr)
            {
                mWeights += lr * mDiffs;
                mWeights.eval();
                mDiffs.reset();
            }
        };
    }
}
