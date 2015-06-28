#ifndef AFML_PARALLEL_HPP_
#define AFML_PARALLEL_HPP_

#include "afml/common.hpp"
#include "afml/container.hpp"

namespace afml {

// https://github.com/facebook/fbcunn/blob/master/fbcunn/AbstractParallel.lua
class AbstractParallel : public Container {
 public:
  AbstractParallel(const size_t dim);
  virtual size_t nextGPU() const;
  // Add the node to run on gpuID
  virtual void add(const size_t gpuID, const NodePtr node);
  virtual NodePtr get(const size_t index) const;
  void asyncCopy(const ArrayPtr source, const ArrayPtr dest);
 protected:
  void distributeGradientWrtOutput();

};

// https://github.com/facebook/fbcunn/blob/master/fbcunn/DataParallel.lua
class DataParallel : public AbstractParallel {
 protected:
  void distributeInput(const ArrayPtr input);
  void gatherGradients();
  void combineGradients(const size_t row, const ArrayPtrVec& gradients);

};

// https://github.com/facebook/fbcunn/blob/master/fbcunn/ModelParallel.lua
class ModelParallel : public AbstractParallel {
 public:
  ModelParallel(const size_t dim);
  virtual size_t nextGPU() const;
  virtual void add(const size_t gpuID, const NodePtr node);
  virtual NodePtr get(const size_t index) const;
  void distributeInput(const ArrayPtr input);

};

}  // namespace afml


#endif /* AFML_PARALLEL_HPP_ */
