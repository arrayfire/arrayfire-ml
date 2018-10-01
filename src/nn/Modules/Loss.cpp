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

        autograd::Variable Loss::operator()(const autograd::Variable &inputs,
                                            const autograd::Variable &targets)
        {
            return this->forward(inputs, targets);
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
            return res;
        }

        static autograd::Variable
        binaryCrossEntropy(const autograd::Variable &inputs,
                           const autograd::Variable &targets)
        {
            return -1 * (targets * log(inputs) + (1 - targets) * log(1 - inputs));
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

        static autograd::Variable
        CrossEntropy(const autograd::Variable &inputs,
                     const autograd::Variable &targets)
        {
            auto correct_idxs  = (range(targets.dims()[0]) + inputs.dims()[0] * targets.array()).as(s32);

            auto exps = exp(inputs);
            auto softmaxScores = exps / tile(sum(exps, {1}), { 1, exps.dims()[1] });

            Variable correct_scores = select_index(softmaxScores, Variable(correct_idxs, false));

            auto losses = -1 * log(correct_scores);
            return losses;
        }

        autograd::Variable CrossEntropyLoss::forward(const autograd::Variable &inputs,
                                                     const autograd::Variable &targets)
        {
            return mean(flat(CrossEntropy(inputs, targets)), {0});
        }

        autograd::Variable CrossEntropyLoss::forward(const autograd::Variable &inputs,
                                                     const autograd::Variable &targets,
                                                     const autograd::Variable &weights)
        {
            return mean(flat(weights * CrossEntropy(inputs, targets)), {0});
        }

        static autograd::Variable
        MarginLoss(const autograd::Variable &inputs,
                const autograd::Variable &targets)
        {
            auto correct_idxs   = (range(targets.dims()[0]) + inputs.dims()[0] * targets.array()).as(s32);
            Variable correct_scores = select_index(inputs, Variable(correct_idxs, false));

            auto scores = inputs - tile(correct_scores, { 1, (int)inputs.dims()[1] } );
            const float margin = 1.f;
            auto losses = max(scores + margin, 0); //gives different results than max(0, scores + margin), "intended" behaviour
            //zero out correct classes, should not affect loss
            losses = set_index(losses, correct_scores, Variable(af::constant(0, correct_scores.dims()[0]), false));
            losses = sum(losses, {1}) / inputs.dims()[1];
            return losses;
        }

        autograd::Variable MultiMarginLoss::forward(const autograd::Variable &inputs,
                                                     const autograd::Variable &targets)
        {
            return mean(flat(MarginLoss(inputs, targets)), {0});
        }

        autograd::Variable MultiMarginLoss::forward(const autograd::Variable &inputs,
                                                     const autograd::Variable &targets,
                                                     const autograd::Variable &weights)
        {
            return mean(flat(weights * MarginLoss(inputs, targets)), {0});
        }
    }
}
