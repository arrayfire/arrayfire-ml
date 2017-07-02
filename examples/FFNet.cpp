/*******************************************************
 * Copyright (c) 2015, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <afml/nn.h>

using namespace af;
using namespace afml;
using namespace afml::nn;

int main()
{
    af::info();
    const int inputSize  = 2;
    const int hiddenSize = 3;
    const int outputSize = 1;
    const int numSamples = 4;
    const double lr = 0.8;

    float hInput[] = {1, 1,
                      0, 0,
                      0, 1,
                      1, 0};

    float hOutput[] = {0,
                       0,
                       1,
                       1};

    af::array in(inputSize, numSamples, hInput);
    af::array out(outputSize, numSamples, hOutput);


    FFNet network(inputSize);
    network.addLinearNode(hiddenSize, 5).addActivationNode();
    network.addLinearNode(outputSize, 5).addActivationNode();

    for (int i = 0; i < 1000; i++) {

        ArrayVector data = network.forward({in});
        double err = af::norm(data[0] - out);

        data[0] = out - data[0];

        if ((i + 1) % 100 == 0) {
            printf("Error at iteration(%d) : %2.10lf\n", i + 1, err);
        }
        network.backward({in}, data);
        network.update(lr);
    }

    af_print(af::round(network.forward({in})[0]));
}
