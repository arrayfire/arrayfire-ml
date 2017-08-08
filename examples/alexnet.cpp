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
    const int numSamples = 1;

    auto in = af::randu(227, 227, 3, numSamples);
    auto out = af::randu(55, 55, 96, 1);

    nn::Sequential alexnet;

    //alexnet.add(nn::Conv2D(11, 11, 4, 4, 0, 0, 3, 96, true));
    alexnet.add(nn::ReLU());

    Variable result;
    for (int i = 0; i < 1000; i++) {
        for (int j = 0; j < numSamples; j++) {
            alexnet.train();
            alexnet.zeroGrad();

            af::array in_j = in(af::span, af::span, af::span, j);
            af::array out_j = out;

            // Forward propagation
            result = alexnet.forward(nn::input(in_j));

            // Calculate loss
            // TODO: Use loss function
            af::array diff = out_j - result.array();

            // Backward propagation
            auto d_result = Variable(diff, false);
            result.backward(d_result);

            // Update parameters
            // TODO: Should use optimizer
            for (auto &param : alexnet.parameters()) {
                param.array() += lr * param.grad().array();
                param.array().eval();
            }
        }

        if ((i + 1) % 100 == 0) {
            alexnet.eval();

            // Forward propagation
            result = alexnet.forward(nn::input(in));

            // Calculate loss
            // TODO: Use loss function
            af::array diff = out - result.array();
            printf("Average Error at iteration(%d) : %lf\n", i + 1, af::mean<float>(af::abs(diff)));
            printf("Predicted\n");
            //af_print(result.array());
            printf("Expected\n");
            //af_print(out);
            printf("\n\n");
        }
    }
    return 0;
}
