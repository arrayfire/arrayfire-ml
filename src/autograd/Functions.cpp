/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <af/autograd/Variable.hpp>
#include <af/autograd/Functions.hpp>

namespace af {
    namespace autograd {

        Variable operator +(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() + rhs.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output);
                inputs[1].addGrad(grad_output);
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

        Variable operator -(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() - rhs.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output);
                inputs[1].addGrad(negate(grad_output));
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

        Variable operator *(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() * rhs.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output * inputs[1]);
                inputs[1].addGrad(grad_output * inputs[0]);
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

        Variable operator /(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() / rhs.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                auto inputs_1_rec = reciprocal(inputs[1]);
                auto grad_input_0 = grad_output * inputs_1_rec;
                inputs[0].addGrad(grad_input_0);
                inputs[1].addGrad(grad_input_0 * negate(inputs[0]) * inputs_1_rec);
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

#define INSTANTIATE_OPERATOR(OP)                                        \
        Variable operator OP(const double &lhs_val, const Variable &rhs) \
        {                                                               \
            auto lhs = Variable(                                        \
                af::constant(lhs_val,                                   \
                             rhs.array().dims(),                        \
                             rhs.array().type()),                       \
                false);                                                 \
            return lhs OP rhs;                                          \
        }                                                               \
        Variable operator OP(const Variable &lhs, const double &rhs_val) \
        {                                                               \
            auto rhs = Variable(                                        \
                af::constant(rhs_val,                                   \
                             lhs.array().dims(), lhs.array().type()),   \
                false);                                                 \
            return lhs OP rhs;                                          \
        }                                                               \

        INSTANTIATE_OPERATOR(+)
        INSTANTIATE_OPERATOR(-)
        INSTANTIATE_OPERATOR(*)
        INSTANTIATE_OPERATOR(/)

#undef INSTANTIATE_OPERATOR

        Variable negate(const Variable &input)
        {
            auto result = 0.0 - input.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(negate(grad_output));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable reciprocal(const Variable &input)
        {
            auto result = 1.0 / input.array();
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                auto res = reciprocal(inputs[0]);
                inputs[0].addGrad(negate(grad_output) * res * res);
            };
            return Variable(result, {input}, grad_func);
        }

        Variable exp(const Variable &input)
        {
            auto result = exp(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output * exp(inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable sin(const Variable &input)
        {
            auto result = sin(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output * cos(inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable cos(const Variable &input)
        {
            auto result = cos(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output * negate(sin(inputs[0])));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable tanh(const Variable &input)
        {
            auto result = tanh(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                auto tmp = tanh(inputs[0]);
                inputs[0].addGrad(grad_output * (1.0 - tmp * tmp));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable sigmoid(const Variable &input)
        {
            auto result = sigmoid(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                auto tmp = sigmoid(inputs[0]);
                inputs[0].addGrad(grad_output * tmp * (1 - tmp));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable transpose(const Variable &input)
        {
            auto result = transpose(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(transpose(grad_output));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable expandAs(const Variable &input, const Variable &reference)
        {
            dim4 dims(1,1,1,1);
            dim4 idims = input.array().dims();
            dim4 rdims = reference.array().dims();
            for (int i = 0; i < 4; i++) {
                dims[i] = rdims[i] / idims[i];
            }
            auto result = tile(input.array(), dims);
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(reduceAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable reduceAs(const Variable &input, const Variable &reference)
        {
            dim4 idims = input.array().dims();
            dim4 rdims = reference.array().dims();
            auto result = input.array();
            for (int i = 0; i < 4; i++) {
                if (idims[i] != rdims[i]) result = sum(result, i);
            }
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(expandAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable matmul(const Variable &lhs, const Variable &rhs)
        {
            // lhs:Input[0] -- [M, N]
            // rhs:Input[1] -- [N, K]
            //matmul(lhs, rhs)
            // -- matmul([M, N], [N, K]) --  [M, K]
            // result:grad_output -- [M, K]
            auto result = matmul(lhs.array(), rhs.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                // matmulNT(grad_output, inputs[1])
                // -- matmulNT([M, K], [N, K])
                // -- matmul([M, K], [K, N]) -- [M, K]
                inputs[0].addGrad(matmulNT(grad_output, inputs[1]));
                // matmulTN(inputs[0], grad_output)
                // -- matmulTN([M, N], [M, K])
                // -- matmul([N, M], [M, K]) -- [N, K]
                inputs[1].addGrad(matmulTN(inputs[0], grad_output));
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

        Variable matmulTN(const Variable &lhs, const Variable &rhs)
        {
            // lhs:Input[0] -- [N, M]
            // rhs:Input[1] -- [N, K]
            // matmulTN(lhs, rhs)
            // -- matmulTN([N, M], [N, K])
            // -- matmul([M, N], [N, K]) -- [M, K]
            // result:grad_output -- [M, K]
            auto result = matmulTN(lhs.array(), rhs.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                // matmulNT(inputs[1], grad_output)
                // -- matmulNT([N, K], [M, K])
                // -- matmul([N, K], [K, M]) -- [N, M]
                inputs[0].addGrad(matmulNT(inputs[1], grad_output));
                // matmul(inputs[0], grad_output)
                // -- matmulNT([N, M], [M, K]) -- [N, K]
                inputs[1].addGrad(matmul(inputs[0], grad_output));
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }

        Variable matmulNT(const Variable &lhs, const Variable &rhs)
        {
            // lhs:Input[0] -- [M, N]
            // rhs:Input[1] -- [K, N]
            // matmulNT(lhs, rhs)
            // -- matmulNT([M, N], [K, N])
            // -- matmul([M, N], [N, K]) -- [M, K]
            // result:grad_output -- [M, K]
            auto result = matmulNT(lhs.array(), rhs.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                // matmul(grad_output, inputs[1])
                // -- matmul([M, K], [K, N]) -- [M, N]
                inputs[0].addGrad(matmul(grad_output, inputs[1]));
                // matmulTN(grad_output, inputs[0])
                // -- matmulTN([M, K], [M, N])
                // -- matmul([K, M], [M, N]) -- [K, N]
                inputs[1].addGrad(matmulTN(grad_output, inputs[0]));
            };
            return Variable(result, {lhs, rhs}, grad_func);
        }
    }
}
