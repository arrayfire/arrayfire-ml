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
#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {
        class Sigmoid : public Module
        {
        public:
            Sigmoid();

            autograd::Variable forward(const autograd::Variable &input);
        };

        class Tanh : public Module
        {
        public:
            Tanh();

            autograd::Variable forward(const autograd::Variable &input);
        };

        class ReLU : public Module
        {
        public:
            ReLU();
         
            autograd::Variable forward(const autograd::Variable &input);
        };
      
        class LeakyReLU : public Module
        {
        private:
            double m_slope;
        public:
            LeakyReLU(double slope = 0.0);
         
            autograd::Variable forward(const autograd::Variable &input);
        };

        class PReLU : public Module
        {
        public:
            PReLU(int size, double spread = 1.0);
            PReLU(const autograd::Variable &w);
         
            autograd::Variable forward(const autograd::Variable &input);
        };

    }
}
