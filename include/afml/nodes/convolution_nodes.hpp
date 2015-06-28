#ifndef AFML_CONVOLUTION_NODES_HPP_
#define AFML_CONVOLUTION_NODES_HPP_

#include "afml/common.hpp"

namespace afml {

// TODO: Which is better, Caffe or Torch7's convolution API?

// https://github.com/BVLC/caffe/blob/master/include/caffe/vision_layers.hpp
class BaseConvolution : public Node {

};

class Convolution : public BaseConvolution {

};

class DeconvolutionLayer : public BaseConvolution {

};


// What's the difference between SpatialConvolution, SpatialConvolutionMM,
// SpatialConvolutionMap,  SpatialFullConvolution,  SpatialFullConvolutionMap,
// TemporalConvolution, and VolumetricConvolution of torch/nn?
// Can they be simplified and unified?
// https://github.com/torch/nn/

// https://github.com/torch/nn/blob/master/SpatialConvolution.lua
class SpatialConvolution : public Node {
 public:
  SpatialConvolution(const string& name, const size_t numInputPlane,
      const size_t numOutputPlane, const size_t kernalWidth,
      const size_t kernalHeight, const size_t strideWidth,
      const size_t strideHeight, const size_t padWidth, const size_t padHeight);
};

}  // namespace afml

#endif /* AFML_CONVOLUTION_NODES_HPP_ */

