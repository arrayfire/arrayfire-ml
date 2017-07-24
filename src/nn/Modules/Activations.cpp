/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd/Functions.hpp>
#include <af/nn/Modules/Activations.hpp>
#include <af/nn/Init.hpp>
namespace af
{
    namespace nn
    {
        using namespace autograd;

        Sigmoid::Sigmoid() {}

        Variable Sigmoid::forward(const Variable &input)
        {
            return sigmoid(input);
        }

        Tanh::Tanh() {}

        Variable Tanh::forward(const Variable &input)
        {
            return tanh(input);
        }

        ReLU::ReLU() {}

        Variable ReLU::forward(const Variable &input)
        {
            return max(input, 0.0);
        }

        LeakyReLU::LeakyReLU(double slope) :
            m_slope(slope)
        {
        }

        Variable LeakyReLU::forward(const Variable &input)
        {
            return max(input, m_slope * input);
        }

        PReLU::PReLU(int size, double value)
        {
            auto w = nn::constant(value, size, 1);
            setParams({w});
        }

        PReLU::PReLU(const Variable &w) :
            Module({w})
        {
        }

        Variable PReLU::forward(const Variable &input)
        {
            auto mask = input >= 0.0;
            return (input * mask) + (input * !mask * tileAs(m_parameters[0], input));
        }

        ELU::ELU(double alpha) :
            m_alpha(alpha)
        {
        }

        Variable ELU::forward(const Variable &input)
        {
            auto mask = input >= 0.0;
            return (mask * input) + (!mask * m_alpha * (exp(input)-1));
        }

        ThresholdReLU::ThresholdReLU(double threshold) :
            m_threshold(threshold)
        {
        }

        Variable ThresholdReLU::forward(const Variable &input)
        {
            auto mask = input >= m_threshold;
            return input * mask;
        }
    }
}
