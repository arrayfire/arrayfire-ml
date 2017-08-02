/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <string>
#include <vector>

#include <af/autograd/Variable.hpp>

namespace af
{
    namespace nn
    {

        class Module
        {
        protected:
            std::vector<autograd::Variable> m_parameters;

            bool m_train;

            Module();

            Module(const std::vector<autograd::Variable> &parameters);

            void setParams(const std::vector<autograd::Variable> &parameters);

        public:

            std::vector<autograd::Variable> parameters();

            void zeroGrad();

            void train();

            void eval();

            virtual autograd::Variable forward(const autograd::Variable &input) = 0;

            autograd::Variable operator()(const autograd::Variable &input);
        };
    }
}
