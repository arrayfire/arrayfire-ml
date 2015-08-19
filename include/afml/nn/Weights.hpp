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

            Weights() : mData(1)
            {
            }


            Weights(int inputSize, int outputSize, float spread) : mData(1)
            {
                mData[0] = af::randu(outputSize, inputSize) * spread - spread / 2; //Weights
            }

            Weights(const af::array &weights) : mData(1)
            {
                mData[0] = weights;
            }

            operator af::array() const
            {
                return mData[0];
            }

            Weights operator+(const Weights &other) const
            {
                return mData[0] + other;
            }

            Weights operator*(const Weights &other) const
            {
                return mData[0] * other;
            }

            Weights operator/(const Weights &other) const
            {
                return mData[0] / other;
            }

            Weights operator-(const Weights &other) const
            {
                return mData[0] - other;
            }

            Weights operator+=(const Weights &other)
            {
                mData[0] += other;
                return *this;
            }

            Weights operator/=(float val)
            {
                mData[0] /= val;
                return *this;
            }

            Weights operator*=(const Weights &other)
            {
                mData[0] *= other;
                return *this;
            }

            Weights operator-=(float val)
            {
                mData[0] -= val;
                return *this;
            }

            void reset()
            {
                mData[0] = af::constant(0, mData[0].dims());
            }

            void eval()
            {
                mData[0].eval();
            }
        };

        Weights operator *(const Weights &lhs, const double &rhs)
        {
            const af::array lhs_arr = lhs;
            return lhs_arr * rhs;
        }

        Weights operator +(const Weights &lhs, const double &rhs)
        {
            const af::array lhs_arr = lhs;
            return lhs_arr + rhs;
        }

        Weights operator /(const Weights &lhs, const double &rhs)
        {
            const af::array lhs_arr = lhs;
            return lhs_arr / rhs;
        }

        Weights operator -(const Weights &lhs, const double &rhs)
        {
            const af::array lhs_arr = lhs;
            return lhs_arr - rhs;
        }

        Weights operator *(const double &lhs, const Weights &rhs)
        {
            const af::array rhs_arr = rhs;
            return lhs * rhs_arr;
        }

        Weights operator +(const double &lhs, const Weights &rhs)
        {
            const af::array rhs_arr = rhs;
            return lhs + rhs_arr;
        }

        Weights operator /(const double &lhs, const Weights &rhs)
        {
            const af::array rhs_arr = rhs;
            return lhs / rhs_arr;
        }

        Weights operator -(const double &lhs, const Weights &rhs)
        {
            const af::array rhs_arr = rhs;
            return lhs - rhs_arr;
        }
    }
}
