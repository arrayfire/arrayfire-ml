/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afml/nn.h>

using namespace afml;
using namespace afml::nn;

int main()
{
    const int inputSize  = 2;
    const int outputSize = 1;
    const int numSamples = 4;
    const double lr = 10;

    float hInput[] = {1, 1,
                      0, 0,
                      1, 0,
                      0, 1};

    float hOutput[] = {1,
                       0,
                       1,
                       1};

    af::array in(inputSize, numSamples, hInput);
    af::array out(outputSize, numSamples, hOutput);

    std::vector<NodePtr> perceptron;
    perceptron.emplace_back(new LinearNode(inputSize, outputSize, 10));
    perceptron.emplace_back(new Sigmoid(inputSize));

    for (int i = 0; i < 10; i++) {
        ArrayVector data = {in};

        std::vector<ArrayVector> inputs(2);
        for (int n = 0; n < 2; n++) {
            inputs[n] = data;
            data = perceptron[n]->forward(data);
        }

        data[0] = out - data[0];

        printf("Error at iteration(%d) : %lf\n", i + 1, af::sum<float>(af::abs(data[0])) / numSamples);

        for (int n = 1; n >= 0; n--) {
            data = perceptron[n]->backward(inputs[n], data);
            perceptron[n]->update(lr);
        }
    }
}
