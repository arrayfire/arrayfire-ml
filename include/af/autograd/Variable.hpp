/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>
#include <unordered_map>

#include <arrayfire.h>

namespace af {
    namespace autograd {
        class Variable
        {
        public:
            typedef std::function<void(std::vector<Variable> &, const Variable &)> GradFunc_t;
            typedef std::unordered_map<std::ptrdiff_t, bool> Cache_t;
            typedef std::vector<Variable> DAG_t;

        private:
            struct Shared {
                Shared();
                Shared(const af::array &data, bool calc_grad);
                Shared(const af::array &data,
                       const std::vector<Variable> &inputs,
                       GradFunc_t grad_func,
                       bool calc_grad);

                bool m_calc_grad;
                af::array m_data;
                std::vector<Variable> m_inputs;
                std::vector<Variable> m_grads;
                GradFunc_t m_grad_func;
            };

            public:

            Variable();
            Variable(const af::array &data, bool calc_grad);
            Variable(const af::array &data,
                     const std::vector<Variable> &inputs,
                     GradFunc_t grad_func);

            af::array array() const;

            Variable grad() const;

            bool isCalcGrad() const;

            void setCalcGrad(bool calc_grad);

            void addGrad(const Variable &child_grad);

            void calcGradInputs(bool retain_grad_graph = false);

            void backward(const Variable &grad, bool retain_grad_graph = false);

            void buildSubGraph(Cache_t &cache, DAG_t &dag);
        private:
            void evalGrad(bool retain_grad_graph = false);

            DAG_t build();
            std::shared_ptr<Shared> m_shared;
        };
    }
}
