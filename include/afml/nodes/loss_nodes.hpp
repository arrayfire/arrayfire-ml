#ifndef AFML_LOSS_NODES_HPP_
#define AFML_LOSS_NODES_HPP_

#include "afml/common.hpp"

namespace afml {

class Accuracy : public Node {

};

// This is probably the most commonly used loss for classification
class NegativeLogLikelihood : public Node {

};

}  // namespace afml


#endif /* AFML_LOSS_NODES_HPP_ */
