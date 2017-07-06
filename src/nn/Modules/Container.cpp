/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd/Variable.hpp>
#include <af/nn/Modules/Container.hpp>

namespace af
{
    namespace nn
    {
        using namespace autograd;

        Container::Container() {}

        ModulePtr Container::get(int id)
        {
            return m_modules[id];
        }

        std::vector<ModulePtr> Container::modules()
        {
            return m_modules;
        }

        Sequential::Sequential() {}

        Variable Sequential::forward(const Variable &input)
        {
            Variable output = input;
            for (auto &module : m_modules) {
                output = module->forward(output);
            }
            return output;
        }
    }
}
