#ifndef AFML_DATA_NODES_HPP_
#define AFML_DATA_NODES_HPP_

#include "afml/common.hpp"

namespace afml {

class Data : public Node {
  virtual ~Data();
  virtual void checkNumNextPrevNodes() {
  }

  virtual void initNode() {
  }

  virtual void forward() {
  }

  virtual void backward() {
  }
};

}  // namespace afml


#endif /* AFML_DATA_NODES_HPP_ */
