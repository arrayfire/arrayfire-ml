namespace cpp afml
namespace csharp afml
namespace go afml
namespace html afml
namespace java afml
namespace js afml
namespace json afml
namespace lua afml
namespace perl afml
namespace php afml
namespace py afml
namespace rb afml

// In Torch7, tensors are backed by storages
struct Storage {
  1: list<i32> dims,
  2: string data
}

struct NodeConfig {
  1: string name,
}