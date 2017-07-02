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

        void backward(Variable var, Variable grad)
        {
            var.addGrad(grad);
            Variable::DAG_t dag = var.build();
            for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
                iter->backward();
            }
        }
    }
    namespace ag = autograd;
}
