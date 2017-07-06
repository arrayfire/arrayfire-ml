/*******************************************************
 * Copyright (c) 2015, ArrayFire
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
    const int numSamples = 4;
    const double lr = 0.005;

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
    for (int i = 0; i < 10; i++) {

        // Forward propagation
        result = perceptron.forward(nn::input(in));

        // Calculate loss
        // TODO: Use loss function
        af::array diff = out - result.array();
        printf("Error at iteration(%d) : %lf\n", i + 1, af::max<float>(af::abs(diff)));

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
    af_print(result.array());
    return 0;
}
