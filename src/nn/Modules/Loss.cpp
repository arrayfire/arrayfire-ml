/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#include <af/autograd/Functions.hpp>
#include <af/nn/Modules/Loss.hpp>


namespace af
{
    namespace nn
    {
        using namespace autograd;

        autograd::Variable Loss::forward(const autograd::Variable &inputs)
        {
            throw af::exception("Loss module requires both inputs and targets");
        }

        autograd::Variable MeanSquaredError::forward(const autograd::Variable &inputs,
                                                     const autograd::Variable &targets)
        {
            auto df = inputs - targets;
            auto res = mean(flat(df * df), {0});
            return res;
        }

        autograd::Variable MeanAbsoluteError::forward(const autograd::Variable &inputs,
                                                      const autograd::Variable &targets)
        {
            auto df = inputs - targets;
            auto res = mean(flat(abs(df)), {0});
        }

        static autograd::Variable
        binaryCrossEntropy(const autograd::Variable &inputs,
                           const autograd::Variable &targets)
        {
            targets * inputs + (1 - targets) * (1 - inputs);
        }

        autograd::Variable BinaryCrossEntropyLoss::forward(const autograd::Variable &inputs,
                                                           const autograd::Variable &targets)
        {
            return mean(flat(binaryCrossEntropy(inputs, targets)), {0});
        }

        autograd::Variable BinaryCrossEntropyLoss::forward(const autograd::Variable &inputs,
                                                           const autograd::Variable &targets,
                                                           const autograd::Variable &weights)
        {
            return mean(flat(weights * binaryCrossEntropy(inputs, targets)), {0});
        }
    }
}
