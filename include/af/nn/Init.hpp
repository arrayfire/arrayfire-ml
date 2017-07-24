/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/autograd/Variable.hpp>

namespace af {
    namespace nn {

        autograd::Variable input(const af::array &arr);

        autograd::Variable parameter(const af::array &arr);

        autograd::Variable uniform(int input_size, int output_size,
                                   double min = 0, double max = 1,
                                   af::dtype type = f32, bool calc_grad=true);

        autograd::Variable uniform(af::dim4 dims,
                                   double min = 0, double max = 1,
                                   af::dtype type = f32, bool calc_grad=true);

        autograd::Variable normal(int input_size, int output_size,
                                  double stdv = 1, double mean = 0,
                                  af::dtype type = f32, bool calc_grad=true);

        autograd::Variable normal(af::dim4 dims,
                                  double stdv = 1, double mean = 0,
                                  af::dtype type = f32, bool calc_grad=true);

        autograd::Variable lecunUniform(int input_size, int output_size,
                                        af::dtype type = f32, bool calc_grad=true);

        autograd::Variable lecunUniform(af::dim4 dims,
                                        af::dtype type = f32, bool calc_grad=true);

        autograd::Variable lecunNormal(int input_size, int output_size,
                                       af::dtype type = f32, bool calc_grad=true);

        autograd::Variable lecunNormal(af::dim4 dims,
                                       af::dtype type = f32, bool calc_grad=true);

        autograd::Variable glorotUniform(int input_size, int output_size,
                                         af::dtype type = f32, bool calc_grad=true);

        autograd::Variable glorotUniform(af::dim4 dims,
                                         af::dtype type = f32, bool calc_grad=true);

        autograd::Variable glorotNormal(int input_size, int output_size,
                                        af::dtype type = f32, bool calc_grad=true);

        autograd::Variable glorotNormal(af::dim4 dims,
                                        af::dtype type = f32, bool calc_grad=true);


        autograd::Variable constant(double val, int input_size, int output_size,
                                    af::dtype type = f32, bool calc_grad=true);

        autograd::Variable constant(double val, af::dim4 dims,
                                    af::dtype type = f32, bool calc_grad=true);

        autograd::Variable identity(int input_size, int output_size,
                                    af::dtype type = f32, bool calc_grad=true);

        autograd::Variable identity(af::dim4 dims,
                                    af::dtype type = f32, bool calc_grad=true);

    }
}
