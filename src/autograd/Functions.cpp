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

        Variable operator >(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() > rhs.array();
            return Variable(result, false);
        }

        Variable operator <(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() < rhs.array();
            return Variable(result, false);
        }

        Variable operator >=(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() >= rhs.array();
            return Variable(result, false);
        }

        Variable operator <=(const Variable &lhs, const Variable &rhs)
        {
            auto result = lhs.array() <= rhs.array();
            return Variable(result, false);
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
        INSTANTIATE_OPERATOR(>)
        INSTANTIATE_OPERATOR(<)
        INSTANTIATE_OPERATOR(>=)
        INSTANTIATE_OPERATOR(<=)

#undef INSTANTIATE_OPERATOR

        Variable operator !(const Variable &input)
        {
            auto result = !input.array();
            return Variable(result, false);
        }

        Variable max(const Variable &lhs, const Variable &rhs)
        {
            auto mask = lhs > rhs;
            auto result = max(lhs.array(), rhs.array());

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad( inputs[2] * grad_output);
                inputs[1].addGrad(!inputs[2] * grad_output);
            };
            return Variable(result, {lhs, rhs, mask}, grad_func);
        }

        Variable min(const Variable &lhs, const Variable &rhs)
        {
            auto mask = lhs < rhs;
            auto result = min(lhs.array(), rhs.array());

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad( inputs[2] * grad_output);
                inputs[1].addGrad(!inputs[2] * grad_output);
            };
            return Variable(result, {lhs, rhs, mask}, grad_func);
        }

#define INSTANTIATE_FUNCTION(FN)                                        \
        Variable FN(const double &lhs_val, const Variable &rhs)         \
        {                                                               \
            auto lhs = Variable(                                        \
                af::constant(lhs_val,                                   \
                             rhs.array().dims(),                        \
                             rhs.array().type()),                       \
                false);                                                 \
            return FN(lhs,rhs);                                         \
        }                                                               \
        Variable FN(const Variable &lhs, const double &rhs_val)         \
        {                                                               \
            auto rhs = Variable(                                        \
                af::constant(rhs_val,                                   \
                             lhs.array().dims(), lhs.array().type()),   \
                false);                                                 \
            return FN(lhs, rhs);                                        \
        }


        INSTANTIATE_FUNCTION(max);
        INSTANTIATE_FUNCTION(min);

#undef INSTANTIATE_FUNCTION

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

        Variable unwrap(const Variable &input, int wx, int wy, int sx, int sy, int px, int py)
        {
            dim4 d = input.array().dims();
            array res = unwrap(input.array(), wx, wy, sx, sy, px, py);
            int params[] = {wx, wy, sx, sy, px, py, (int)d[0], (int)d[1]};
            auto tmp = Variable(array(8, params), false);

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                int* t = inputs[1].array().host<int>();
                inputs[0].addGrad(wrap(grad_output, t[6], t[7], t[0], t[1], t[2], t[3], t[4], t[5]));
            };
            return Variable(res, {input, tmp}, grad_func);
        }

        Variable wrap(const Variable &input, int ox, int oy, int wx, int wy, int sx, int sy, int px, int py)
        {
            array res = wrap(input.array(), ox, oy, wx, wy, sx, sy, px, py);
            int params[] = {wx, wy, sx, sy, px, py};
            auto tmp = Variable(array(6, params), false);

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                int* t = inputs[1].array().host<int>();
                inputs[0].addGrad(unwrap(grad_output, t[0], t[1], t[2], t[3], t[4], t[5]));
            };
            return Variable(res, {input, tmp}, grad_func);
        }

        Variable moddims(const Variable &input, int d0, int d1, int d2, int d3)
        {

            dim4 orig = input.array().dims();
            if(d1 == -1) d1 = orig[1];
            if(d2 == -1) d2 = orig[2];
            if(d3 == -1) d3 = orig[3];
            auto res = moddims(input.array(), d0, d1, d2, d3);

            int params[] = {(int)orig[0], (int)orig[1], (int)orig[2], (int)orig[3]};
            auto tmp = Variable(array(4, params), false);

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output){
                int *p = inputs[1].array().host<int>();
                inputs[0].addGrad(moddims(grad_output, p[0], p[1], p[2], p[3]));
            };
            return Variable(res, {input, tmp}, grad_func);
        }

        Variable reorder(const Variable &input, int d0, int d1, int d2, int d3)
        {
            array res = reorder(input.array(), d0, d1, d2, d3);

            int tmp[] = {d0, d1, d2, d3};
            int tmp2[4];
            for(int i = 0; i < 4; i++){
                tmp2[tmp[i]] = i;
            }
            auto reverse = Variable(array(4, tmp2), false);

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output){
                int *r = inputs[1].array().host<int>();
                inputs[0].addGrad(reorder(grad_output, r[0], r[1], r[2], r[3]));
            };
            return Variable(res, {input, reverse}, grad_func);
        }

        Variable conv2d(const Variable &input, const Variable &weights, int wx, int wy, int sx, int sy, int px, int py)
        {
            dim4 idims = input.array().dims();      // (x_i, y_i, c_i,  n  )
            dim4 wdims = weights.array().dims();    // (wx,  wy,  c_i,  c_o)

            int x_i = idims[0];                     //size of x dim of input
            int y_i = idims[1];                     //size of y dim of input
            int c_i = idims[2];                     //number of input channels
            int n   = idims[3];                     //batch size (1 for now)

            int x_o = (x_i + 2 * px - wx) / sx + 1; //size of x dim of output
            int y_o = (y_i + 2 * py - wy) / sy + 1; //size of x dim of output
            int c_o = wdims[3];                     //number of output channels

            array windows = unwrap(input.array(), wx, wy, sx, sy, px, py);

            array lhs = moddims(
                reorder(windows, 1, 0, 2, 3),
                x_o * y_o, wx * wy * c_i, n, 1);
            array rhs = moddims(weights.array(), wx * wy * c_i, c_o, 1, 1);

            //TODO: This loop can be replaced with a batched matmult as soon as
            //that is added to arrayfire
            std::vector<array> out;
            for(int i = 0; i < n; i++){
                array res = matmul(lhs(span, span, i), rhs);
                out.push_back(moddims(res , x_o, y_o, c_o, 1));
            }

            //LOL @ C++ API
            array result = out[0];
            for(int i = 1; i < n; i+=3){
                int rem = n - i;
                if(rem >= 3){
                    result = join(3, result, out[i], out[i+1], out[i+2]);
                }else if(rem == 2){
                    result = join(3, result, out[i], out[i+1]);
                    break;
                }else if(rem == 1){
                    result = join(3, result, out[i]);
                    break;
                }else{
                    break;
                }
            }

            int tmp[] = {wx, wy, sx, sy, px, py, c_i, x_o, y_o};
            auto params = Variable(array(9, tmp), false);

            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                int* p = inputs[3].array().host<int>(); //unpack parameters
                dim4 odims = grad_output.array().dims();
                dim4 wdims = inputs[1].array().dims();
                dim4 idims = inputs[0].array().dims();

                auto grad_out_reshape = moddims(grad_output, odims[0]*odims[1], odims[2], odims[3], 1);

                auto weights_reshape = moddims(inputs[1], wdims[0]*wdims[1]*wdims[2], wdims[3], 1, 1);

                //TODO: This really needs batched matmul...
                //TODO: This doesn't work for n > 1
                //TODO: Can these lines be shortened? - This seems like a large grad function - perhaps this
                // could all be implemented in Conv2D::forward(). I had to implement the helper functions anyways
                auto a = matmulNT(grad_out_reshape, weights_reshape);
                auto adims = a.array().dims();
                auto b = moddims(a, adims[0], p[0]*p[1], p[6], adims[3]);
                auto c = reorder(b, 1, 0, 2, 3);
                inputs[0].addGrad(wrap(c, idims[0], idims[1], p[0], p[1], p[2], p[3], p[4], p[5]));

                auto d = matmulTN(inputs[2],grad_out_reshape);
                inputs[1].addGrad(moddims(d, p[0], p[1], p[6], d.array().dims()[1]));

                /*
                  for(int i = 0; i < odims[3]; i++){
                  inputs[0].addGrad(wrap(), p[0], p[1], p[2], p[3], p[4], p[5]);
                  inputs[0].addGrad(wrap(matmulNT(Variable(lhs(span, span, span, i), false), p_tmp), p[0], p[1], p[2], p[3], p[4], p[5]));
                  inputs[1].addGrad(matmulTN(inputs[0], Variable(lhs(span, span, span, i), true)));
                  }
                */
            };
            return Variable(result, {input, weights, Variable(lhs, false), params}, grad_func);

        }
    }
}
