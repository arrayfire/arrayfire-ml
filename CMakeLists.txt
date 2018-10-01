cmake_minimum_required(VERSION 3.5.1)

project(ArrayFireML
  VERSION 0.1.0
  LANGUAGES C CXX)

find_package(ArrayFire REQUIRED)

add_library(afml SHARED "")

target_sources(afml
  PRIVATE
  src/autograd/Functions.cpp
  src/autograd/Variable.cpp
  src/nn/Modules/Activations.cpp
  src/nn/Modules/Container.cpp
  src/nn/Modules/Linear.cpp
  src/nn/Modules/Loss.cpp
  src/nn/Modules/Module.cpp
  src/nn/Modules/Dropout.cpp
  src/nn/Init.cpp
  src/optim/Optimizers.cpp
  )

target_include_directories(afml
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(afml PUBLIC ArrayFire::af)

set_target_properties(afml
  PROPERTIES
  VERSION "${ArrayFireML_VERSION}"
  SOVERSION "${ArrayFireML_VERSION_MAJOR}"
  CXX_STANDARD 11)

add_subdirectory(examples)

option(PACKAGE_TESTS "Build tests" ON)
if(PACKAGE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
