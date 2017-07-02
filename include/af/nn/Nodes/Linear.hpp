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
#include <af/nn/Weights.hpp>
#include <af/nn/Nodes/Node.hpp>

namespace af
{
    namespace nn
    {
        class LinearNode : public Node
        {
        private:

            Weights mWeight, mBias;
            Weights mWeightDiff, mBiasDiff;

        public:

            LinearNode(const int inputSize, const int outputSize,
                       float spread = 0.05,
                       const char *name="none") :
                Node(1, &inputSize, 1, &outputSize, name),
                mWeight(inputSize, outputSize, spread),
                mBias(1, outputSize, spread),
                mWeightDiff(), mBiasDiff()
            {
            }

            ArrayVector forward(const ArrayVector &input)
            {
                return {af::matmul(mWeight, input[0]) +
                        af::tile(mBias, 1, input[0].dims(1))};
            }

            ArrayVector backward(const ArrayVector &input,
                                 const ArrayVector &gradOutput)
            {
                float m = input[0].dims(1);

                mWeightDiff = af::matmulNT(gradOutput[0], input[0]) / m;
                mBiasDiff = af::sum(gradOutput[0], 1) / m;

                return { af::matmulTN(mWeight, gradOutput[0]) };
            }

            void update(float lr)
            {
                mWeight += lr * mWeightDiff;
                mBias   += lr * mBiasDiff;

                mWeight.eval();
                mBias.eval();

                mWeightDiff.reset();
                mBiasDiff.reset();
            }
        };
    }
}
