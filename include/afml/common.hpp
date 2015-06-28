#ifndef AFML_COMMON_HPP_
#define AFML_COMMON_HPP_

#include <arrayfire.h>

#if __cplusplus < 201100L
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#else
#include <initializer_list>
#include <memory>
#include <unordered_map>
#include <cstddef>
#endif

#include "afml/thrift/afml_types.h"

namespace afml {

using namespace af;
using namespace boost;
using namespace std;

#if __cplusplus < 201100L
using boost::make_shared;
using boost::shared_ptr;
using boost::nullptr;
using boost::unordered_map;
#else
using std::initializer_list;
using std::make_shared;
using std::shared_ptr; // Can CUDA with this?
using std::nullptr;
using std::unordered_map;
#endif

typedef unordered_map map;

// Just to be consistent in camel case style.
typedef array Array;

typedef vector<Array> ArrayVec;
typedef shared_ptr<Array> ArrayPtr;
typedef vector<ArrayPtr> ArrayPtrVec;

class Node;
typedef shared_ptr<Node> NodePtr;
typedef vector<NodePtr> NodePtrVec;

}  // namespace afml

#endif /* AFML_COMMON_HPP_ */
