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
#include <arrayfire.h>

#include <vector>

namespace af
{
    namespace optim
    {

        class Optimizer
        {
        protected:
            std::vector<autograd::Variable> m_parameters;
        public:

            Optimizer(const std::vector<autograd::Variable> &parameters);

            virtual void update() = 0;

            void zeroGrad();
        };

        class SGDOptimizer : public Optimizer
        {
            bool m_use_nesterov;
            double m_lr;
            double m_mu;
            double m_wd;
            std::vector<af::array> m_velocities;
        public:
            SGDOptimizer(const std::vector<autograd::Variable> &parameters,
                         double learning_rate, double momentum = 0,
                         double weight_decay = 0,
                         bool use_nesterov = false);
            void update();
        };

        class AdamOptimizer : public Optimizer
        {
            double m_lr;
            double m_beta1;
            double m_beta2;
            double m_eps;
            double m_wd;
            int m_count;
            std::vector<af::array> m_biased_first;
            std::vector<af::array> m_biased_second;
        public:
            AdamOptimizer(const std::vector<autograd::Variable> &parameters,
                          double learning_rate,
                          double beta1 = 0.9,
                          double beta2 = 0.999,
                          double epsilon = 1E-8,
                          double weight_decay = 0);
            void update();
        };

        class RMSPropOptimizer : public Optimizer
        {
            bool m_use_first;
            double m_lr;
            double m_rho;
            double m_eps;
            double m_wd;
            std::vector<af::array> m_first;
            std::vector<af::array> m_second;
        public:
            RMSPropOptimizer(const std::vector<autograd::Variable> &parameters,
                             double learning_rate,
                             double rho = 0.99,
                             double epsilon = 1E-8,
                             double weight_decay = 0,
                             bool use_first = false);
            void update();
        };

    }
}
