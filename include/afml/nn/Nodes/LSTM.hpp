/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <string>
#include <afml/util/common.hpp>
#include <afml/nn/Weights.hpp>
#include <afml/nn/Nodes/Node.hpp>

namespace afml
{
    namespace nn
    {
        class LSTMNode : public Node
        {
        private:
            /* Standard LSTM Implementation
             * Tutorial: http://deeplearning.net/tutorial/lstm.html
             * Reference: http://www.cs.toronto.edu/~graves/phd.pdf */
					  Weights mWi, mUi, mIDiffs;   // input gate weights + recurrence
            Weights mWf, mUf, mFDiffs;   // forget gate weights + recurrence
            Weights mWc, mUc, mCDiffs;   // memory cell weights + recurrence
            Weights mWo, mUo, mODiffs;   // output gate weights + recurrence

            enum DataIndex{
                DATA = 0,
                PREVIOUS_HIDDEN,
                PREVIOUS_MEMORY,
            };

        public:

            LSTMNode(const int inputSize, const int outputSize,
                     std::string outer_activation = "tanh",
                     std::string inner_activation = "hard_sigmoid",
                     float outerW_activation = 0.05,
                     float innerW_activation = 0.05,
                     const char *name="none") :
                Node(1, &inputSize, 1, &outputSize, name),
                mWi(inputSize, outputSize, outerW_activation),
                mUi(outputSize, outputSize, innerW_activation),
                mWf(inputSize, outputSize, outerW_activation),
                mUf(outputSize, outputSize, innerW_activation),
                mWc(inputSize, outputSize, outerW_activation),
                mUc(outputSize, outputSize, innerW_activation),
                mWo(inputSize, outputSize, outerW_activation),
                mUo(outputSize, outputSize, innerW_activation),
                mIDiffs(), mFDiffs(), mCDiffs(), mODiffs()
            {
                mWi.setBias(af::constant(0.0f, mWi.getBias().dims()));
                // forget gate NEEDS to be one: http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
                mWf.setBias(af::constant(1.0f, mWi.getBias().dims()));
                mWc.setBias(af::constant(0.0f, mWi.getBias().dims()));
                mWo.setBias(af::constant(0.0f, mWi.getBias().dims()));
            }

            ArrayVector forward(const ArrayVector &input)
            {
                //TODO: Refactor to initializations
                // i_t, f_t, o_t = inner_activation
                // C_t, h_t = outer_activation
                af::array i_t = af::tanh(af::matmul(mWi.getWeights(), input[DataIndex::DATA]) +
                                         af::matmul(mUi.getWeights(), input[DataIndex::PREVIOUS_HIDDEN]) +
                                         af::tile(mWi.getBias(), 1, input[DataIndex::DATA].dims(1)));
                af::array Cprime_t = af::tanh(af::matmul(mWc.getWeights(), input[DataIndex::DATA]) +
                                         af::matmul(mUc.getWeights(), input[DataIndex::PREVIOUS_HIDDEN]) +
                                         af::tile(mWc.getBias(), 1, input[DataIndex::DATA].dims(1)));
                af::array f_t = af::tanh(af::matmul(mWc.getWeights(), input[DataIndex::DATA]) +
                                         af::matmul(mUc.getWeights(), input[DataIndex::PREVIOUS_HIDDEN]) +
                                         af::tile(mWc.getBias(), 1, input[DataIndex::DATA].dims(1)));
                af::array c_t = af::dot(i_t, Cprime_t) + af::dot(f_t, input[DataIndex::PREVIOUS_MEMORY]);
                af::array o_t = af::tanh(af::matmul(mWo.getWeights(), input[DataIndex::DATA]) +
                                         af::matmul(mUo.getWeights(), input[DataIndex::PREVIOUS_HIDDEN]) +
                                         af::tile(mWo.getBias(), 1, input[DataIndex::DATA].dims(1)));
                af::array h_t = af::dot(o_t, af::tanh(c_t));
                return {h_t, c_t};
            }

            ArrayVector backward(const ArrayVector &input,
                                 const ArrayVector &gradOutput)
            {
                throw std::runtime_error("Backward for LSTM not implemented yet");
            }

            void update(float lr)
            {
                mWi += lr * mIDiffs;
                mWf += lr * mFDiffs;
                mWc += lr * mCDiffs;
                mWo += lr * mODiffs;

                mWi.eval(); mUi.eval();
                mWf.eval(); mUf.eval();
                mWc.eval(); mUc.eval();
                mWo.eval(); mUo.eval();

                mIDiffs.reset();
                mFDiffs.reset();
                mCDiffs.reset();
                mODiffs.reset();
            }
        };
    }
}
