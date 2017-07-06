/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd.h>
#include <af/nn.h>

using namespace af;
using namespace af::nn;
using namespace af::autograd;

int main()
{
    const int inputSize  = 2;
    const int outputSize = 1;
    const double lr = 0.1;
    const int numSamples = 4;

    float hInput[] = {1, 1,
                      0, 0,
                      1, 0,
                      0, 1};

    float hOutput[] = {1,
                       0,
                       1,
                       1};

    auto in = af::array(inputSize, numSamples, hInput);
    auto out = af::array(outputSize, numSamples, hOutput);

    nn::Sequential perceptron;

    perceptron.add(nn::Linear(inputSize, outputSize));
    perceptron.add(nn::Sigmoid());

    Variable result;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < numSamples; j++) {
            perceptron.train();
            perceptron.zeroGrad();

            af::array in_j = in(af::span, j);
            af::array out_j = out(af::span, j);

            // Forward propagation
            result = perceptron.forward(nn::input(in_j));

            // Calculate loss
            // TODO: Use loss function
            af::array diff = out_j - result.array();

            // Backward propagation
            auto d_result = Variable(diff, false);
            result.backward(d_result);

            // Update parameters
            // TODO: Should use optimizer
            for (auto param : perceptron.parameters()) {
                param.array() += lr * param.grad().array();
                param.array().eval();
            }
        }

        if ((i + 1) % 100 == 0) {
            perceptron.eval();

            // Forward propagation
            result = perceptron.forward(nn::input(in));

            // Calculate loss
            // TODO: Use loss function
            af::array diff = out - result.array();
            printf("Average Error at iteration(%d) : %lf\n", i + 1, af::mean<float>(af::abs(diff)));
            printf("Predicted\n");
            af_print(result.array());
            printf("Expected\n");
            af_print(out);
            printf("\n\n");
        }
    }
    return 0;
}
