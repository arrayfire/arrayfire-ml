/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd/Variable.hpp>
#include <af/autograd/Functions.hpp>

namespace af {
    namespace autograd {

        Variable::Shared::Shared() :
            m_calc_grad(true),
            m_data(),
            m_inputs(),
            m_grads(),
            m_grad_func(nullptr)
        {}

        Variable::Shared::Shared(af::array data, bool calc_grad) :
            m_calc_grad(calc_grad),
            m_data(data),
            m_inputs(),
            m_grads(),
            m_grad_func(nullptr)
        {}

        Variable::Shared::Shared(af::array data,
                                 std::vector<Variable> inputs,
                                 GradFunc_t grad_func,
                                 bool calc_grad) :
            m_calc_grad(calc_grad),
            m_data(data),
            m_inputs(inputs.begin(), inputs.end()),
            m_grads(),
            m_grad_func(grad_func)
        {}

        Variable::Variable() :
            m_shared(new Shared())
        {
        }

        Variable::Variable(af::array data, bool calc_grad) :
            m_shared(new Shared(data, calc_grad))
        {}

        Variable::Variable(af::array data,
                 std::vector<Variable> inputs,
                 GradFunc_t grad_func) :
            m_shared(nullptr)
        {
            bool calc_grad = false;
            for (auto input : inputs) {
                calc_grad |= input.isCalcGrad();
            }
            if (calc_grad) {
                m_shared = std::shared_ptr<Shared>(new Shared(data, inputs, grad_func, true));
            } else {
                m_shared = std::shared_ptr<Shared>(new Shared(data, false));
            }
        }

        af::array Variable::array() const
        {
            return m_shared->m_data;
        }

        Variable Variable::grad() const
        {
            if (!m_shared->m_calc_grad) {
                throw af::exception("Gradient calclation disabled.");
            }
            if (m_shared->m_grads.size() == 0) {
                throw af::exception("Gradient hasn't been calculated yet.");
            }
            return m_shared->m_grads[0];
        }

        bool Variable::isCalcGrad()
        {
            return m_shared->m_calc_grad;
        }

        void Variable::setCalcGrad(bool calc_grad)
        {
            m_shared->m_calc_grad = calc_grad;
            if (!calc_grad) {
                m_shared->m_grad_func = nullptr;
                m_shared->m_inputs.clear();
                m_shared->m_grads.clear();
            }
        }

        void Variable::addGrad(Variable child_grad)
        {
            if (m_shared->m_calc_grad) {
                m_shared->m_grads.push_back(child_grad);
            }
        }

        void Variable::evalGrad()
        {
            // Flag asking not to calculate gradients
            if (!m_shared->m_calc_grad) return;
            Variable grad = m_shared->m_grads[0];
            for (unsigned i = 1; i < m_shared->m_grads.size(); i++) {
                grad = grad + m_shared->m_grads[i];
            }
            grad.array().eval();
            m_shared->m_grads.clear();
            m_shared->m_grads.push_back(grad);
        }

        void Variable::calcGradInputs()
        {
            evalGrad();
            if (m_shared->m_grad_func) {
                m_shared->m_grad_func(m_shared->m_inputs, m_shared->m_grads[0]);
            }
        }

        void Variable::backward(Variable grad)
        {
            this->addGrad(grad);
            Variable::DAG_t dag = this->build();
            for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
                iter->calcGradInputs();
            }
        }

        Variable::DAG_t Variable::build()
        {
            Cache_t cache;
                    Variable::DAG_t dag;
            this->buildSubGraph(cache, dag);
            return dag;
        }

        void Variable::buildSubGraph(Cache_t &cache, Variable::DAG_t &dag)
        {
            std::ptrdiff_t id = (std::ptrdiff_t)m_shared.get();
            if (cache.find(id) != cache.end()) {
                return;
            }
            for (auto input : m_shared->m_inputs) {
                input.buildSubGraph(cache, dag);
            }
            cache[id] = true;
            dag.push_back(*this);
        }
    }
}
