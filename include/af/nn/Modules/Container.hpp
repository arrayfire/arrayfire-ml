/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <memory>
#include <af/autograd/Variable.hpp>
#include <af/nn/Modules/Module.hpp>

namespace af
{
    namespace nn
    {

        typedef std::shared_ptr<Module> ModulePtr;

        class Container : public Module
        {
        protected:

            std::vector<ModulePtr> m_modules;

            Container();

        public:

            template<typename T>
            void add(T module)
            {
                m_modules.emplace_back(new T(module));
                for (auto param : module.parameters()) {
                    m_parameters.push_back(param);
                }
            }

            ModulePtr get(int id);

            std::vector<ModulePtr> modules();

            virtual autograd::Variable forward(const autograd::Variable &input) = 0;
        };

        class Sequential : public Container
        {
        public:

            Sequential();

            virtual autograd::Variable forward(const autograd::Variable &input);
        };
    }
}
