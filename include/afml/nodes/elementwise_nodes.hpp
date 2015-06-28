#ifndef AFML_ELEMENTWISE_NODES_HPP_
#define AFML_ELEMENTWISE_NODES_HPP_

#include "afml/common.hpp"

namespace afml {

class ReLU : public Node {
  virtual ~ReLU();
  virtual void checkNumNextPrevNodes() {
    CHECK(nextNodes_.size() >= expectedMinNumNextNodes_);
    CHECK(prevNodes_.size() == expectedNumPrevNodes_);
  }

  virtual void initNode() {
    output_->resize(prevNodes_[0].output()->dims());
  }

  virtual void forward() {
    output_ = prevNodes_[0].output() * (prevNodes_[0].output() > 0);
  }

  virtual void backward() {
    gradient_ = output_ > 0;
    computeGradientInput();
    composeGradient();
  }
};

}  // namespace afml


#endif /* AFML_ELEMENTWISE_NODES_HPP_ */
