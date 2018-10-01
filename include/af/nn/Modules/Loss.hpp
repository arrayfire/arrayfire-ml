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
        class Loss : public Module
        {
        public:
            Loss() {}

            virtual autograd::Variable forward(const autograd::Variable &inputs,
                                               const autograd::Variable &targets) = 0;

            autograd::Variable forward(const autograd::Variable &inputs);

            autograd::Variable operator()(const autograd::Variable &inputs,
                                          const autograd::Variable &targets);
        };

        class MeanSquaredError : public Loss
        {
        public:
            MeanSquaredError() {}

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets);
        };

        class MeanAbsoluteError : public Loss
        {
        public:
            MeanAbsoluteError() {}

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets);
        };

        class BinaryCrossEntropyLoss : public Loss
        {
        public:
            BinaryCrossEntropyLoss() {}

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets);

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets,
                                       const autograd::Variable &weights);
        };

        class CrossEntropyLoss : public Loss
        {
        public:
            CrossEntropyLoss() {}

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets);

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets,
                                       const autograd::Variable &weights);
        };

        class MultiMarginLoss : public Loss
        {
        public:
            MultiMarginLoss() {}

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets);

            autograd::Variable forward(const autograd::Variable &inputs,
                                       const autograd::Variable &targets,
                                       const autograd::Variable &weights);
        };

        typedef MeanSquaredError MSE;
        typedef MeanAbsoluteError MAE;
        typedef MeanAbsoluteError L1Loss;
        typedef BinaryCrossEntropyLoss BCELoss;
        typedef CrossEntropyLoss CELoss;
    }
}
