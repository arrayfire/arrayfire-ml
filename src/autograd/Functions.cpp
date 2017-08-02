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

        Variable log(const Variable &input)
        {
            auto result = log(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(grad_output / inputs[0]);
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

        Variable tileAs(const Variable &input, const Variable &reference)
        {
            dim4 dims(1,1,1,1);
            dim4 rdims = reference.dims();
            dim4 idims = input.dims();
            for (int i = 0; i < 4; i++) {
                dims[i] = rdims[i] / idims[i];
            }
            auto result = tile(input.array(), dims);
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(sumAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable sumAs(const Variable &input, const Variable &reference)
        {
            dim4 rdims = reference.dims();
            dim4 idims = input.dims();
            auto result = input.array();
            for (int i = 0; i < 4; i++) {
                if (idims[i] != rdims[i]) result = sum(result, i);
            }
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(tileAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable tile(const Variable &input, const std::vector<int> &repeats)
        {
            dim4 dims;
            for (size_t i = 0; i < repeats.size(); i++) {
                dims[i] = repeats[i];
            }
            auto result = tile(input.array(), dims);
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(sumAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable sum(const Variable &input, const std::vector<int> &axes)
        {
            auto result = input.array();
            for (size_t i = 0; i < axes.size(); i++) {
                result = sum(result, axes[i]);
            }
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(tileAs(grad_output, inputs[0]));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable mean(const Variable &input, const std::vector<int> &axes)
        {
            auto result = input.array();
            for (size_t i = 0; i < axes.size(); i++) {
                result = mean(result, axes[i]);
            }
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                dim4 odims = grad_output.dims();
                dim4 idims = inputs[0].dims();
                dim_t count = 1;
                for (int i = 0; i < 4; i++) {
                    count *= idims[i] / odims[i];
                }
                inputs[0].addGrad(count * tileAs(grad_output, inputs[0]));
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

        Variable abs(const Variable &input)
        {
            auto result = af::abs(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                // af::sign returns signbit
                // Convert it into -1, 1
                auto sign = Variable(1 - 2 * af::sign(inputs[0].array()), false);
                inputs[0].addGrad(sign * grad_output);
            };
            return Variable(result, {input}, grad_func);
        }

        Variable flat(const Variable &input)
        {
            auto result = af::flat(input.array());
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(moddims(grad_output, inputs[0].dims()));
            };
            return Variable(result, {input}, grad_func);
        }

        Variable moddims(const Variable &input, const dim4 &dims)
        {
            auto result = af::moddims(input.array(), dims);
            auto grad_func = [](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(moddims(grad_output, inputs[0].dims()));
            };
            return Variable(result, {input}, grad_func);
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

            auto grad_func = [tmp2](std::vector<Variable> &inputs, const Variable &grad_output){
                inputs[0].addGrad(reorder(grad_output, tmp2[0], tmp2[1], tmp2[2], tmp2[3]));
            };
            return Variable(res, {input, reverse}, grad_func);
        }

        Variable unwrap(const Variable &input, int wx, int wy, int sx, int sy, int px, int py)
        {
            array res = unwrap(input.array(), wx, wy, sx, sy, px, py);
            auto grad_func = [wx, wy, sx, sy, px, py](std::vector<Variable> &inputs, const Variable &grad_output) {
                dim4 d = inputs[0].dims();
                inputs[0].addGrad(wrap(grad_output, d[0], d[1], wx, wy, sx, sy, px, py));
            };
            return Variable(res, {input}, grad_func);
        }

        Variable wrap(const Variable &input, int ox, int oy, int wx, int wy, int sx, int sy, int px, int py)
        {
            array res = wrap(input.array(), ox, oy, wx, wy, sx, sy, px, py);
            auto grad_func = [wx, wy, sx, sy, px, py](std::vector<Variable> &inputs, const Variable &grad_output) {
                inputs[0].addGrad(unwrap(grad_output, wx, wy, sx, sy, px, py));
            };
            return Variable(res, {input}, grad_func);
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
                dim4(x_o * y_o, wx * wy * c_i, n, 1));
            array rhs = moddims(weights.array(), dim4(wx * wy * c_i, c_o, 1, 1));

            //TODO: This loop can be replaced with a batched matmult as soon as
            //that is added to arrayfire
            std::vector<array> out;
            for(int i = 0; i < n; i++){
                array res = matmul(lhs(span, span, i), rhs);
                out.push_back(moddims(res , dim4(x_o, y_o, c_o, 1)));
            }

            //LOL @ C++ API - need this loop to have arbitrary batch size
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

            auto grad_func = [wx, wy, sx, sy, px, py, c_i, n](std::vector<Variable> &inputs, const Variable &grad_output) {
                dim4 odims = grad_output.array().dims();
                dim4 wdims = inputs[1].array().dims();
                dim4 idims = inputs[0].array().dims();

                auto grad_out_reshape = moddims(grad_output, dim4(odims[0]*odims[1], odims[2], odims[3], 1));

                auto weights_reshape = moddims(inputs[1], dim4(wdims[0]*wdims[1]*wdims[2], wdims[3], 1, 1));

                //TODO: This really needs batched matmul...
                //TODO: This doesn't work for n > 1
                //TODO: Can these lines be shortened? - This seems like a large grad function - perhaps this
                // could all be implemented in Conv2D::forward(). I had to implement the helper functions anyways
                /*
                std::vector<array> out;
                for(int i = 0; i < n; i++){
                auto a = matmulNT(grad_out_reshape(span, span, i), weights_reshape); //Problem is here - can't call () on Variable
                auto adims = a.array().dims();
                auto b = moddims(a, dim4(adims[0], wx*wy, c_i, adims[3]));
                auto c = reorder(b, 1, 0, 2, 3);
                out.push_pack(wrap(c, idims[0], idims[1], wx, wy, sx, sy, px, py));
                }

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
                */
                auto a = matmulNT(grad_out_reshape, weights_reshape);
                auto adims = a.array().dims();
                auto b = moddims(a, dim4(adims[0], wx*wy, c_i, adims[3]));
                auto c = reorder(b, 1, 0, 2, 3);
                inputs[0].addGrad(wrap(c, idims[0], idims[1], wx, wy, sx, sy, px, py));

                auto d = matmulTN(inputs[2],grad_out_reshape);
                inputs[1].addGrad(moddims(d, dim4(wx, wy, c_i, d.dims()[1])));

                /*
                  for(int i = 0; i < odims[3]; i++){
                  inputs[0].addGrad(wrap(), p[0], p[1], p[2], p[3], p[4], p[5]);
                  inputs[0].addGrad(wrap(matmulNT(Variable(lhs(span, span, span, i), false), p_tmp), p[0], p[1], p[2], p[3], p[4], p[5]));
                  inputs[1].addGrad(matmulTN(inputs[0], Variable(lhs(span, span, span, i), true)));
                  }
                */

            };
            return Variable(result, {input, weights, Variable(lhs, false)}, grad_func);

        }



    }
}
