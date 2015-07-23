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

namespace afml
{
    namespace nn
    {
        class Weights
        {
            ArrayVector mData;

        public:

            Weights() : mData(2)
            {
            }


            Weights(int inputSize, int outputSize, double spread) : mData(2)
            {
                mData[0] = af::randu(outputSize, inputSize) * spread - spread / 2; //Weights
                mData[1] = af::randu(outputSize,         1) * spread - spread / 2; //Biases
            }

            Weights(const af::array &weights, const af::array &bias) : mData(2)
            {
                mData[0] = weights;
                mData[1] = bias;
            }

            af::array getWeights() const
            {
                return mData[0];
            }

            af::array getBias() const
            {
                return mData[1];
            }

            void setWeights(const af::array &weights)
            {
                mData[0] = weights;
            }

            void setBias(const af::array &bias)
            {
                mData[1] = bias;
            }

            Weights operator+=(const Weights &other)
            {
                mData[0] += other.getWeights();
                mData[1] += other.getBias();
                return *this;
            }

            Weights operator/=(double val)
            {
                mData[0] /= val;
                mData[1] /= val;
                return *this;
            }

            void reset()
            {
                mData[0] = af::constant(0, mData[0].dims());
                mData[1] = af::constant(0, mData[1].dims());
            }

            void eval()
            {
                mData[0].eval();
                mData[1].eval();
            }
        };

        Weights operator*(const double val, const Weights W)
        {
            return Weights(val * W.getWeights(), val * W.getBias());
        }

    }
}
