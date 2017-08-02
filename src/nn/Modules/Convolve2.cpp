/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/autograd/Functions.hpp>
#include <af/nn/Init.hpp>
#include <af/nn/Modules/Convolve2.hpp>
//output will be ho x wo x no x n
namespace af
{
    namespace nn
    {
        using namespace autograd;

        Convolve2::Convolve2(int wx, int wy, int sx, int sy, int px, int py, int n_in, int n_out, bool bias) :
            m_wx(wx),
            m_wy(wy),
            m_sx(sx),
            m_sy(sy),
            m_px(px),
            m_py(py),
            m_bias(bias)
        {
            auto w = nn::lecunNormal(dim4(wx, wy, n_in, n_out));
            if (bias) {
                auto b = nn::lecunNormal(dim4(1, 1, n_out, 1));
                setParams({w, b});
            } else {
                setParams({w});
            }
        }

        Convolve2::Convolve2(const Variable &w, int sx, int sy, int px, int py) :
            m_sx(sx),
            m_sy(sy),
            m_px(px),
            m_py(py),
            m_bias(false),
            Module({w})
        {
            dim4 pdims = w.dims();
            m_wx = pdims[0];
            m_wy = pdims[1];
        }

        Convolve2::Convolve2(const Variable &w, const Variable &b, int sx, int sy, int px, int py) :
            m_sx(sx),
            m_sy(sy),
            m_px(px),
            m_py(py),
            m_bias(true),
            Module({w, b})
        {
            if (b.dims()[1] != 1) {
                throw af::exception("nn::Linear: Bias must be a vector.");
            }
            dim4 pdims = w.dims();
            m_wx = pdims[0];
            m_wy = pdims[1];
        }

        Variable Convolve2::forward(const Variable &input)
        {
            auto res = convolve2(input, m_parameters[0], m_wx, m_wy, m_sx, m_sy, m_px, m_py);
            if (m_bias) {
                res = res + tileAs(m_parameters[1], res);
            }
            return res;
        }
    }
}
