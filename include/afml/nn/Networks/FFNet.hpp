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
#include <afml/nn/Nodes/Linear.hpp>
#include <afml/nn/Activations.hpp>

namespace afml
{
    namespace nn
    {
        class FeedForwardNetwork : public Node
        {
        private:
            IntVector mNodeSizes;
            std::vector<NodePtr> mNodes;
            std::vector<ArrayVector> mData;

        public:

            FeedForwardNetwork(const int inputSize, const char *name="none") :
                Node(1, &inputSize, name),
                mNodeSizes(1),
                mNodes(0),
                mData(0)
            {
                mNodeSizes[0] = inputSize;
            }

            FeedForwardNetwork& addLinearNode(const int size, const double spread = 0.05)
            {
                int num = (int)mNodeSizes.size();

                mNodes.emplace_back(new LinearNode(mNodeSizes[num - 1], size, spread));
                mNodeSizes.push_back(size);

                this->setOutSizes(1, &size);
                return *this;
            }

            template<typename ActivationType = SigmoidNode>
            FeedForwardNetwork& addActivationNode()
            {
                int size = (int)mNodeSizes.back();

                // Ensure ActivationType is derived from ActivationNode
                ActivationNode *node = new ActivationType(size);
                mNodeSizes.push_back(size);
                mNodes.emplace_back(node);
                return *this;
            }

            template<typename NodeType>
            FeedForwardNetwork& addNode(const NodeType &node)
            {
                mNodes.emplace_back(new NodeType(node));

                // TODO: Throw exception of node.getOutSizes() has >1 length
                int size = node.getOutSizes()[0];

                mNodeSizes.push_back(size);
                return *this;
            }

            ArrayVector forward(const ArrayVector &input)
            {
                mData.resize(mNodeSizes.size());
                mData[0] = input;
                for (int i = 0; i < (int)mNodes.size(); i++) {
                    mData[i + 1] = mNodes[i]->forward(mData[i]);
                }
                return mData.back();
            }

            ArrayVector backward(const ArrayVector &input,
                                 const ArrayVector &gradOutput)
            {
                //TODO: Assert input coming is same as the stored input
                ArrayVector currGradOutput = gradOutput;
                for (int i = (int)mNodes.size() - 1; i >= 0; i--) {
                    currGradOutput = mNodes[i]->backward(mData[i], currGradOutput);
                }
                return currGradOutput;
            }

            void update(double lr)
            {
                for(int i = 0; i < (int)mNodes.size(); i++) {
                    mNodes[i]->update(lr);
                }
            }
        };

        typedef FeedForwardNetwork FFNet;
    }
}
