#ifndef AFML_CONTAINER_HPP_
#define AFML_CONTAINER_HPP_

#include "afml/afml.hpp"

namespace afml {

// From Torch7
// https://github.com/torch/nn/blob/master/Container.lua
// Contains multiple nodes for easier management or building complex networks
class Container : public Node {
 public:
  Container();

  // Return this shared_ptr to chain calls add()->add()->add()
  virtual shared_ptr<Container> add(const NodePtr node);

  // Returns the contained modules at index index.
  virtual NodePtr get(const size_t index) const;

  // Returns the number of contained modules.
  size_t size() const;

  virtual void forward();
  virtual void backward();
  virtual void toString() const;
};

// https://github.com/torch/nn/blob/master/Concat.lua
class Concat : public Container {
 public:
  Concat(const size_t concatDim);
  virtual ~Concat();

};

// https://github.com/torch/nn/blob/master/Sequential.lua
// To simplify management of sequentially connected nodes.
class Sequential : public Container {
  virtual ~Sequential();
  // Return this shared_ptr to chain calls add()->add()->add()
  virtual shared_ptr<SequentialContainer> add(const NodePtr node);
  virtual void insert(const NodePtr node, const size_t index);
  virtual void remove (const size_t index);
  virtual void toString() const;
};

// https://github.com/torch/nn/blob/master/Parallel.lua
// To run multiple copies of a part of a model on different GPUs.
class Parallel : public Container {
 public:
  Parallel(const size_t inputDim, const size_t outputDim);
  virtual ~Parallel();
  virtual void forward();
  virtual void backward();
  virtual void toString() const;
};


}  // namespace afml

#endif /* AFML_CONTAINER_HPP_ */
