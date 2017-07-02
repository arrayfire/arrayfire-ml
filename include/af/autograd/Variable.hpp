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
            typedef std::function<void(std::vector<Variable>, Variable)> BackwardFunc_t;
            typedef std::unordered_map<std::ptrdiff_t, bool> Cache_t;
            typedef std::vector<Variable> DAG_t;

        private:
            class Shared {
            public:
                Shared() :
                    m_data(),
                    m_grad(),
                    m_inputs(),
                    m_grad_parts(),
                    m_backward(nullptr)
                {}

                Shared(af::array data) :
                    m_data(data),
                    m_grad(af::constant(0, data.dims(), data.type())),
                    m_inputs(),
                    m_grad_parts(),
                    m_backward(nullptr)
                {}

                Shared(af::array data, std::vector<Variable> inputs, BackwardFunc_t backward) :
                    m_data(data),
                    m_grad(af::constant(0, data.dims(), data.type())),
                    m_inputs(inputs.begin(), inputs.end()),
                    m_grad_parts(),
                    m_backward(backward)
                {}

                af::array getData() const
                {
                    return m_data;
                }

                af::array getGrad() const
                {
                    return m_grad;
                }

                void addGrad(Variable grad)
                {
                    m_grad_parts.push_back(grad);
                }

                std::vector<Variable> getGradParts()
                {
                    return m_grad_parts;
                }

                std::vector<Variable> getInputs()
                {
                    return m_inputs;
                }

                void evalGrad()
                {
                    m_grad = m_grad_parts[0].getData();
                    for (int i = 1; i < (int)m_grad_parts.size(); i++) {
                        m_grad += m_grad_parts[i].getData();
                    }
                    af::eval(m_grad);
                }

                void backward()
                {
                    this->evalGrad();
                    if (m_backward) m_backward(m_inputs, m_grad);
                }

            private:
                af::array m_data;
                af::array m_grad;
                std::vector<Variable> m_inputs;
                std::vector<Variable> m_grad_parts;
                BackwardFunc_t m_backward;
            };

            public:

            Variable() :
                m_shared(new Shared())
            {
            }

            Variable(af::array data) :
                m_shared(new Shared(data))
            {}

            Variable(af::array data,
                     std::vector<Variable> inputs,
                     BackwardFunc_t backward) :
                m_shared(new Shared(data, inputs, backward))
            {}

            af::array getData() const
            {
                return m_shared->getData();
            }

            af::array getGrad() const
            {
                return m_shared->getGrad();
            }

            void addGrad(Variable child_grad)
            {
                m_shared->addGrad(child_grad);
            }

            std::vector<Variable> getInputs() const
            {
                return m_shared->getInputs();
            }

            void evalGrad()
            {
                m_shared->evalGrad();
            }

            void backward()
            {
                m_shared->backward();
            }

            DAG_t build()
            {
                Cache_t cache;
                DAG_t dag;
                this->buildGraph(cache, dag);
                return dag;
            }

            void buildGraph(Cache_t &cache, DAG_t &dag)
            {
                std::ptrdiff_t id = (std::ptrdiff_t)m_shared.get();
                if (cache.find(id) != cache.end()) {
                    return;
                }
                for (auto input : m_shared->getInputs()) {
                    input.buildGraph(cache, dag);
                }
                cache[id] = true;
                dag.push_back(*this);
            }
        private:
            std::shared_ptr<Shared> m_shared;
        };
    }
    namespace ag = autograd;
}
