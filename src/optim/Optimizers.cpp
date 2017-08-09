/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/optim/Optimizers.hpp>

#include <cmath>

using af::autograd::Variable;
using std::vector;

// References:
// SGD and Momentum: http://cs231n.github.io/neural-networks-3/#sgd
// Adam: https://arxiv.org/pdf/1412.6980.pdf
// RMSProp: https://arxiv.org/pdf/1308.0850v5.pdf

// Comparision between various update rules:
// https://www.quora.com/What-are-differences-between-update-rules-like-AdaDelta-RMSProp-AdaGrad-and-AdaM

namespace af
{
    namespace optim
    {
        Optimizer::Optimizer(const vector<Variable> &parameters)
            : m_parameters(parameters.begin(), parameters.end())
        {
        }

        void Optimizer::zeroGrad()
        {
            for (auto &parameter : m_parameters) {
                parameter.zeroGrad();
            }
        }

        SGDOptimizer::SGDOptimizer(const vector<Variable> &parameters,
                                   double learning_rate, double momentum,
                                   double weight_decay, bool use_nesterov)
            : Optimizer(parameters),
              m_use_nesterov(use_nesterov),
              m_lr(learning_rate),
              m_mu(momentum),
              m_wd(weight_decay),
              m_velocities()
        {
            if (momentum != 0) {
                m_velocities.reserve(parameters.size());
                for (const auto &parameter : m_parameters) {
                    m_velocities.push_back(af::constant(0, parameter.dims(), parameter.type()));
                    m_velocities.back().eval();
                }
            }
        }

        void SGDOptimizer::update()
        {
            for (size_t i = 0; i < m_parameters.size(); i++) {

                const af::array &grad = m_parameters[i].grad().array();
                af::array &data = m_parameters[i].array();

                if (m_wd != 0) {
                    // Weight decay term
                    data = data - m_wd * data;
                }

                if (m_mu != 0) {
                    af::array &velocity = m_velocities[i];

                    // Regular momentum
                    velocity = m_mu * velocity - m_lr * grad;
                    if (m_use_nesterov) {
                        // Update for nesterov momentum
                        data = data + velocity * m_mu  - m_lr * grad;
                    } else {
                        data = data + velocity;
                    }

                    af::eval(velocity, data);
                } else {

                    data = data - m_lr * grad;
                    af::eval(data);
                }
            }
        }


        AdamOptimizer::AdamOptimizer(const vector<Variable> &parameters,
                                     double learning_rate,
                                     double beta1, double beta2,
                                     double epsilon, double weight_decay)
            : Optimizer(parameters),
              m_lr(learning_rate),
              m_beta1(beta1),
              m_beta2(beta2),
              m_eps(epsilon),
              m_wd(weight_decay),
              m_count(0),
              m_biased_first(),
              m_biased_second()
        {
            m_biased_first.reserve(parameters.size());
            m_biased_second.reserve(parameters.size());

            for (const auto &parameter : m_parameters) {
                m_biased_first.push_back(af::constant(0, parameter.dims(), parameter.type()));
                m_biased_second.push_back(af::constant(0, parameter.dims(), parameter.type()));

                m_biased_first.back().eval();
                m_biased_second.back().eval();
            }
        }

        void AdamOptimizer::update()
        {
            for (size_t i = 0; i < m_parameters.size(); i++) {
                const af::array &grad = m_parameters[i].grad().array();
                af::array &data = m_parameters[i].array();

                if (m_wd != 0) {
                    // Weight decay term
                    data = data - m_wd * data;
                }

                af::array &biased_first = m_biased_first[i];
                af::array &biased_second = m_biased_second[i];

                biased_first  = m_beta1 * biased_first  + (1 - m_beta1) * grad;
                biased_second = m_beta2 * biased_second + (1 - m_beta2) * grad * grad;

                m_count++;

                double corrected_bias1 = 1 - std::pow(m_beta1, m_count);
                double corrected_bias2 = 1 - std::pow(m_beta2, m_count);
                double corrected_lr = m_lr * std::sqrt(corrected_bias2) / corrected_bias1;

                data = data - (corrected_lr * biased_first) / (af::sqrt(biased_second) + m_eps);

                af::eval(data, biased_first, biased_second);
            }
        }

        RMSPropOptimizer::RMSPropOptimizer(const vector<Variable> &parameters,
                                           double learning_rate,
                                           double rho,
                                           double epsilon,
                                           double weight_decay,
                                           bool use_first)
            : Optimizer(parameters),
              m_use_first(use_first),
              m_lr(learning_rate),
              m_rho(rho),
              m_eps(epsilon),
              m_wd(weight_decay),
              m_first(),
              m_second()
        {
            if (m_use_first) m_first.reserve(parameters.size());
            m_second.reserve(parameters.size());

            for (const auto &parameter : m_parameters) {
                if (m_use_first) {
                    m_first.push_back(af::constant(0, parameter.dims(), parameter.type()));
                    m_first.back().eval();
                }

                m_second.push_back(af::constant(0, parameter.dims(), parameter.type()));
                m_second.back().eval();
            }
        }

        void RMSPropOptimizer::update()
        {
            for (size_t i = 0; i < m_parameters.size(); i++) {
                const af::array &grad = m_parameters[i].grad().array();
                af::array &data = m_parameters[i].array();

                if (m_wd != 0) {
                    // Weight decay term
                    data = data - m_wd * data;
                }

                af::array &second = m_second[i];
                second = m_rho * second + (1 - m_rho) * grad * grad;

                // Create shallow copy of second so that we don't update "second" below
                af::array moments = second;
                if (m_use_first) {
                    af::array &first = m_first[i];
                    first  = m_rho * first  + (1 - m_rho) * grad;
                    moments = moments - first * first;
                }

                data = data - (m_lr * grad) / (af::sqrt(moments) + m_eps);

                if (m_use_first) {
                    af::array &first = m_first[i];
                    af::eval(data, first, second);
                } else {
                    af::eval(data, second);
                }
            }
        }
    }
}
