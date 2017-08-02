/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Convolve2 : public Module
        {
        private:
            bool m_bias;
            int m_wx;
            int m_wy;
            int m_sx;
            int m_sy;
            int m_px;
            int m_py;
        public:
            Convolve2(int wx, int wy, int sx, int sy, int px, int py, int n_in, int n_out, bool bias = true);

            Convolve2(const autograd::Variable &w, int sx = 1, int sy = 1, int px = 0, int py = 0);

            Convolve2(const autograd::Variable &w, const autograd::Variable &b, int sx = 1, int sy = 1, int px = 0, int py = 0);

            autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
