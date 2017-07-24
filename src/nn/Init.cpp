/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <cmath>

#include <af/nn/Init.hpp>

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

        autograd::Variable uniform(int output_size, int input_size,
                                   double min, double max,
                                   af::dtype type, bool calc_grad)
        {
            return nn::uniform(af::dim4(output_size, input_size), min, max, type, calc_grad);
        }

        autograd::Variable uniform(af::dim4 dims, double min, double max,
                                   af::dtype type, bool calc_grad)
        {
            af::array result = af::randu(dims, type);
            if (min != 0 || max != 1) {
                result = (max - min) * result + min;
            }
            return Variable(result, calc_grad);
        }

        autograd::Variable normal(int output_size, int input_size,
                                  double stdv, double mean,
                                  af::dtype type, bool calc_grad)
        {
            return nn::normal(af::dim4(output_size, input_size), stdv, mean, type, calc_grad);
        }

        autograd::Variable normal(af::dim4 dims, double stdv, double mean,
                                  af::dtype type, bool calc_grad)
        {
            af::array result = af::randn(dims, type);
            if (mean != 0 || stdv != 1) {
                result = stdv * result + mean;
            }
            return Variable(result, calc_grad);
        }

        autograd::Variable lecunUniform(int output_size, int input_size,
                                        af::dtype type, bool calc_grad)
        {
            return nn::lecunUniform(af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable lecunUniform(af::dim4 dims,
                                        af::dtype type, bool calc_grad)
        {
            dim_t elements = dims.elements();
            dim_t fan_in = elements / dims[1];
            double stdv = ::sqrt(1.0/(double)fan_in);
            double limit = ::sqrt(3.0) * stdv;
            return nn::uniform(dims, -limit, limit, type, calc_grad);
        }

        autograd::Variable lecunNormal(int output_size, int input_size,
                                       af::dtype type, bool calc_grad)
        {
            return nn::lecunNormal(af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable lecunNormal(af::dim4 dims,
                                       af::dtype type, bool calc_grad)
        {
            dim_t elements = dims.elements();
            dim_t fan_in = elements / dims[1];
            double stdv = ::sqrt(1.0/(double)fan_in);
            return nn::normal(dims, 0, stdv, type, calc_grad);
        }

        autograd::Variable glorotUniform(int output_size, int input_size,
                                         af::dtype type, bool calc_grad)
        {
            return nn::glorotUniform(af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable glorotUniform(af::dim4 dims,
                                         af::dtype type, bool calc_grad)
        {
            dim_t elements = dims.elements();
            dim_t fan_in = elements / dims[1];
            dim_t fan_out = elements / dims[0];
            double stdv = ::sqrt(2.0/(double)(fan_in + fan_out));
            double limit = ::sqrt(3.0) * stdv;
            return nn::uniform(dims, -limit, limit, type, calc_grad);
        }

        autograd::Variable glorotNormal(int output_size, int input_size,
                                        af::dtype type, bool calc_grad)
        {
            return nn::glorotNormal(af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable glorotNormal(af::dim4 dims,
                                        af::dtype type, bool calc_grad)
        {
            dim_t elements = dims.elements();
            dim_t fan_in = elements / dims[1];
            dim_t fan_out = elements / dims[0];
            double stdv = ::sqrt(2.0/(double)(fan_in + fan_out));
            return nn::normal(dims, 0, stdv, type, calc_grad);
        }

        autograd::Variable constant(double val, int output_size, int input_size,
                                    af::dtype type, bool calc_grad)
        {
            return nn::constant(val, af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable constant(double val, af::dim4 dims,
                                    af::dtype type, bool calc_grad)
        {
            return Variable(af::constant(val, dims, type), calc_grad);
        }

        autograd::Variable identity(int output_size, int input_size,
                                    af::dtype type, bool calc_grad)
        {
            return nn::identity(af::dim4(output_size, input_size), type, calc_grad);
        }

        autograd::Variable identity(af::dim4 dims,
                                    af::dtype type, bool calc_grad)
        {
            return Variable(af::identity(dims, type), calc_grad);
        }
    }
}
