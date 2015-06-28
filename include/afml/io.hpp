#ifndef AFML_IO_HPP_
#define AFML_IO_HPP_

#include "afml/common.hpp"

namespace afml {

class SerDe {
 public:

  // Defined in thrift/afml.thrift
  Data& serialize(const Array& arr) const;
  Array& deserialize(const Data& data);
};


}  // namespace afml


#endif /* AFML_IO_HPP_ */
