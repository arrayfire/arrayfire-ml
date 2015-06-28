#ifndef AFML_NODE_HPP_
#define AFML_NODE_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "afml/common.hpp"

namespace afml {

// https://github.com/BVLC/caffe/blob/master/include/caffe/layer.hpp
// https://github.com/torch/nn/blob/master/Module.lua
class Node {
 public:
  explicit Node(const NodeConfig& nodeConfig);
  virtual ~Node();
  // getAllNodes and traverse are from
  // https://github.com/zxie/nn/blob/master/nets/graph.py
  static NodePtrVec getAllNodes(const NodePtrVec& startNodes);
  template<class Function>
  static void traverse(const NodePtrVec& startNodes, Function fn) {
    NodePtrVec readyNodes;
    map<string, int> deps;
    for (size_t i = 0; i, startNodes.size(); ++i) {
      if (startNodes[i]->numPrev() == 0) {
        readyNodes.push_back(startNodes[i]);
      }
      deps[startNodes[i]->name()] = startNodes[i]->numPrev();
    }
    vector < string > names;
    while (readyNodes.size() > 0) {
      NodePtrVec nextReadyNodes;
      for (size_t i = 0; i < readyNodes.size(); ++i) {
        fn(readyNodes[i]);
        names = readyNodes[i]->nextNames();
        for (size_t j = 0; j < names.size(); ++j) {
          deps[names[j]]--;
          if (deps[names[j]] == 0) {
            nextReadyNodes.push_back(readyNodes[i]->next(names[j]));
          }
        }
        deps.erase(readyNodes[i]->name());
      }
      readyNodes = nextReadyNodes;
    }
  }

  // Input and output are more general than the top and bottom of Caffe
  virtual void forward() = 0;
  // Since the network is DAG, propagate_back is more general than
  // propagate_down of Caffe
  virtual void backward() = 0;

  void computeGradientWrtOutput() {
    gradientWrtInput_ = nextNodes_[0].gradient();
    for (size_t i = 1; i < nextNodes_.size(); ++i) {
      gradientWrtInput_ += nextNodes_[i].gradient();
    }
  }

  void composeGradient() {
    gradientWrtInput_ *= gradientWrtOutput_;
  }

  // Must be called after all the addNext has been called
  // i.e. the nodes in the network has connected with each other
  // Calls initNode which subclass can override
  void init() {
    thisNode_.reset(NodePtr<this>);
    checkNumNextPrevNodes();
    initNode();
  }

  string name() const {
    return name_;
  }

  // Must make sure there's no copy to return Array
  ArrayPtr output() const {
    return output_;
  }

  ArrayPtr gradient() const {
    return gradientWrtInput();
  }

  ArrayPtr gradientWrtInput() const {
    return gradientWrtInput_;
  }

  ArrayPtr gradientWrtOutput() const {
    return gradientWrtOutput_;
  }

  ArrayPtr param(const string& name) const {
    CHECK(params_.find(name) != params_.end());
    return params_[name];
  }

  ArrayPtr gradientWrtParams(const string& name) const {
    CHECK(gradientWrtParams_.find(name) != gradientWrtParams_.end());
    return gradientWrtParams_[name];
  }

  void add(NodePtr node) {
    addNext(node);
  }

  // The model is DAG(Directed Acyclic Graph)
  void addNext(NodePtr node) {
    if (!hasNext(node) && node->hasPrev(thisNode_)) {
      next_[node->name()] = node;
      nextNodes_.push_back(node);
      node->addPrev(thisNode_);
    }
  }

  void addPrev(NodePtr node) {
    prev_[node->name()] = node;
    prevNodes_.push_back(node);
  }

// Shortcut to add multiple nodes
  void addNext(vector<NodePtr>& nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      addNext(nodes[i]);
    }
  }

  void addPrev(vector<NodePtr>& nodes) {
    for (size_t i = 0; i < nodes.size(); ++i) {
      addPrev(nodes[i]);
    }
  }

  bool hasNext(NodePtr node) const {
    return next_.find(node) != next_.end();
  }

  bool hasPrev(NodePtr node) const {
    return prev_.find(node) != prev_.end();
  }

  map<string, NodePtr>& next() const {
    return next_;
  }

  map<string, NodePtr>& prev() const {
    return prev_;
  }

  vector<string> nextNames() const {
    vector < string > names;
    for (map<string, NodePtr>::iterator iter = next_.begin();
        iter != next_.end(); ++iter) {
      names.push_back(iter->name());
    }
    return names;
  }

  vector<string> prevNames() const {
    vector < string > names;
    for (map<string, NodePtr>::iterator iter = prev_.begin();
        iter != prev_.end(); ++iter) {
      names.push_back(iter->name());
    }
    return names;
  }

  vector<NodePtr>& nextNodes() const {
    return nextNodes_;
  }

  vector<NodePtr>& prevNodes() const {
    return prevNodes_;
  }

  NodePtr next(const string& name) const {
    if (next_.find(name) != next_.end()) {
      return next_[name];
    }
    return nullptr;
  }

  NodePtr prev(const string& name) const {
    if (prev_.find(name) != prev_.end()) {
      return prev_[name];
    }
    return nullptr;
  }

  size_t numNext() const {
    return next_.size();
  }

  size_t numPrev() const {
    return prev_.size();
  }

  // From Torch7 module API
  // https://github.com/torch/nn/blob/master/doc/module.md
  void training() {
    train_ = true;
  }
  void evaluating() {
    train_ = false;
  }

  void shareParams(const NotePtr node);
  void shareParams(const NotePtr node, const string& name);
  void shareParams(const NotePtr node, const string& name1, const string& name2);
  void shareParams(const NotePtr node, const vector<string>& names);
  NodePtr clone();
  NodePtr clone(const string& name);
  NodePtr clone(const string& name1, const string& name2);
  NodePtr clone(const vector<string>& names);

  // Use CPU or GPU
  void cpu();
  void gpu();

  virtual void toString() const;

 protected:
  virtual void checkNumNextPrevNodes() = 0;
  virtual void initNode() = 0;

  string name_;
  NodeConfig& config_;
  NodePtr thisNode_;
  ArrayPtr output_;
  // wrt = ith regardt to
  ArrayPtr gradientWrtInput_;
  ArrayPtr gradientWrtOutput_;
  map<string, ArrayPtr> params_;
  map<string, ArrayPtr> gradientWrtParams_;
  map<string, NodePtr> next_;
  vector<NodePtr> nextNodes_;
  map<string, NodePtr> prev_;
  vector<NodePtr> prevNodes_;
  size_t expectedNumNextNodes_;
  size_t expectedNumPrevNodes_;
  size_t expectedMinNumNextNodes_;
  size_t expectedMinNumPrevNodes_;
  size_t expectedMaxNumNextNodes_;
  size_t expectedMaxNumPrevNodes_;
};

}  // namespace afml

#endif /* AFML_NODE_HPP_ */
