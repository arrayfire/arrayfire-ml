#ifndef AFML_COMMON_NODES_HPP_
#define AFML_COMMON_NODES_HPP_

#include "afml/common.hpp"

namespace afml {

enum PoolingType {
  AVERAGE,
  MAX,
  STOCHASTIC
};

// Fully connected inner product of input and output
class Linear : public Node {
  virtual ~Linear();
  virtual void checkNumNextPrevNodes() {
    CHECK(nextNodes_.size() >= expectedMinNumNextNodes_);
    CHECK(prevNodes_.size() == expectedNumPrevNodes_);
  }

  virtual void initNode() {
    output_->resize(prevNodes_[0].output()->dims());
  }

  virtual void forward() {
    output_ = params_['W'] * prevNodes_[0].output();
  }

  virtual void backward() {
    computeGradientWrtOutput();
    gradientWrtInput_ = params_['W']->T() * gradientWrtOutput_;
    gradientWrtParams_['W'] = gradientWrtOutput_->matmul(output_->T());
  }
};

class Softmax : public Node {
  virtual ~Softmax();
  virtual void checkNumNextPrevNodes() {
    CHECK(nextNodes_.size() >= expectedMinNumNextNodes_);
    CHECK(prevNodes_.size() == expectedNumPrevNodes_);
  }

  virtual void initNode() {
    output_->resize(prevNodes_[0].output()->dims());
  }

  virtual void forward() {
    int axis = 0;
    Array probs  = (prevNodes_[0].output() - prevNodes_[0].output().max(axis)).exp();
    output_ = probs / probs.sum(axis);
  }

  virtual void backward() {
    gradient_ = output_ * output_ - output_;
  }

};

}  // namespace afml


#endif /* AFML_COMMON_NODES_HPP_ */
