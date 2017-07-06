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

        Variable::Shared::Shared(const af::array &data, bool calc_grad) :
            m_calc_grad(calc_grad),
            m_data(data),
            m_inputs(),
            m_grads(),
            m_grad_func(nullptr)
        {}

        Variable::Shared::Shared(const af::array &data,
                                 const std::vector<Variable> &inputs,
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

        Variable::Variable(const af::array &data, bool calc_grad) :
            m_shared(new Shared(data, calc_grad))
        {}

        Variable::Variable(const af::array &data,
                           const std::vector<Variable> &inputs,
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

        std::ptrdiff_t Variable::id() const
        {
            return (std::ptrdiff_t)m_shared.get();
        }

        std::vector<Variable> Variable::getInputs() const
        {
            return m_shared->m_inputs;
        }

        bool Variable::isCalcGrad() const
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

        void Variable::addGrad(const Variable &child_grad)
        {
            if (m_shared->m_calc_grad) {
                m_shared->m_grads.push_back(child_grad);
            }
        }

        void Variable::evalGrad(bool retain_grad_graph)
        {
            // Flag asking not to calculate gradients
            if (!m_shared->m_calc_grad) return;

            // Best not to evaluate the JIT immediately if theres only a single gradient
            Variable grad = m_shared->m_grads[0];
            if (m_shared->m_grads.size() > 1) {
                for (unsigned i = 1; i < m_shared->m_grads.size(); i++) {
                    grad = grad + m_shared->m_grads[i];
                }
                grad.array().eval();
                m_shared->m_grads.resize(1);
            }

            // Remove the graph if not needed
            if (!retain_grad_graph) {
                // This can be done by extracting af::array and ignoring everything else
                auto grad_data = grad.array();
                // Since there's no graph leading this, set calc_grad to false
                grad = Variable(grad_data, false);
            }

            m_shared->m_grads[0] = grad;
        }

        void Variable::calcGradInputs(bool retain_grad_graph)
        {
            evalGrad();
            if (m_shared->m_grad_func) {
                m_shared->m_grad_func(m_shared->m_inputs, m_shared->m_grads[0]);
            }
        }

        void Variable::backward(const Variable &grad, bool retain_grad_graph)
        {
            this->addGrad(grad);
            Variable::DAG_t dag = Variable::build(*this);
            for (auto iter = dag.rbegin(); iter != dag.rend(); iter++) {
                iter->calcGradInputs(retain_grad_graph);
            }
        }

        Variable::DAG_t Variable::build(const Variable &var)
        {
            Cache_t cache;
            Variable::DAG_t dag;
            Variable::buildSubGraph(cache, dag, var);
            return dag;
        }

        void Variable::buildSubGraph(Cache_t &cache, Variable::DAG_t &dag, const Variable &var)
        {
            std::ptrdiff_t id = var.id();
            if (cache.find(id) != cache.end()) {
                return;
            }
            for (auto input : var.getInputs()) {
                Variable::buildSubGraph(cache, dag, input);
            }
            cache[id] = true;
            dag.push_back(var);
        }
    }
}
