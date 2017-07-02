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
    namespace autograd {

        Variable operator +(const Variable lhs, const Variable rhs)
        {
            auto result = lhs.getData() + rhs.getData();
            auto backward = [](std::vector<Variable> inputs, Variable grad_output) {
                inputs[0].addGrad(grad_output);
                inputs[1].addGrad(grad_output);
            };
            return Variable(result, {lhs, rhs}, backward);
        }

        Variable operator *(const Variable lhs, const Variable rhs)
        {
            auto result = lhs.getData() * rhs.getData();
            auto backward = [](std::vector<Variable> inputs, Variable grad_output) {
                inputs[0].addGrad(grad_output * inputs[1]);
                inputs[1].addGrad(grad_output * inputs[0]);
            };
            return Variable(result, {lhs, rhs}, backward);
        }

    }
    namespace ag = autograd;
}
