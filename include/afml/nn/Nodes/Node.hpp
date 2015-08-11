/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <afml/util/common.hpp>
#include <afml/nn/Weights.hpp>

#include <memory>
#include <cstring>

namespace afml
{
    namespace nn
    {

        class Node
        {
        private:
            IntVector  mInputSizes;
            IntVector mOutputSizes;

            char mName[MAX_NAME_SIZE];

            void set(const int *inputSizes, const int *outputSizes,
                     const char *name, const int count)
            {
                for (int i = 0; i <  (int)mInputSizes.size(); i++)  mInputSizes[i] =  inputSizes[i];
                for (int i = 0; i < (int)mOutputSizes.size(); i++) mOutputSizes[i] = outputSizes[i];

                int len = std::min(count, MAX_NAME_SIZE - 1);
                std::memcpy(mName, name, len);
                mName[len] = 0;
            }

        protected:
            void setOutSizes(const int numOutputs, const int *outputSizes)
            {
                mOutputSizes.resize(numOutputs);
                for (int i = 0; i < numOutputs; i++) {
                    mOutputSizes[i] = outputSizes[i];
                }
            }

            Node(const int numInputs, const int *inputSizes, const char *name):
                mInputSizes(numInputs), mOutputSizes(numInputs)
            {
                set(inputSizes, inputSizes, name, (int)strlen(name));
            }

        public:

            Node(const int numInputs, const int *inputSizes,
                 const int numOutputs, const int *outputSizes, const char *name)
                : mInputSizes(numInputs), mOutputSizes(numOutputs)
            {
                set(inputSizes, outputSizes, name, (int)strlen(name));
            }

            Node(const std::vector<int> &inputSizes,
                 const std::vector<int> &outputSizes,
                 const std::string &name)
                : mInputSizes((int)inputSizes.size()), mOutputSizes((int)outputSizes.size())
            {
                set(&inputSizes[0], &outputSizes[0], name.c_str(), (int)name.size());
            }

            virtual ArrayVector forward(const ArrayVector &input)
            {
                return input;
            }

            virtual ArrayVector backward(const ArrayVector &input,
                                         const ArrayVector &gradOutput)
            {
                return gradOutput;
            }

            virtual void update(float lr) {}

            //TODO: Add a method that actually returns this information to the user
            virtual void info()
            {
                std::cout << "Name: "  << mName << std::endl;
                std::cout << "Input sizes: " << std::endl;

                for (int i = 0; i <  (int)mInputSizes.size(); i++) {
                    std::cout << mInputSizes[i] << std::endl;
                }

                std::cout << "Output sizes: " << std::endl;
                for (int i = 0; i < (int)mOutputSizes.size(); i++) {
                    std::cout << mOutputSizes[i] << std::endl;
                }
            }

            IntVector getInSizes() const
            {
                return mInputSizes;
            }

            IntVector getOutSizes() const
            {
                return mOutputSizes;
            }
        };

        typedef std::shared_ptr<Node> NodePtr;
    }
}
