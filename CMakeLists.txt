cmake_minimum_required(VERSION 3.5.1)

project(ArrayFireML
  VERSION 0.1.0
  LANGUAGES C CXX)

find_package(ArrayFire REQUIRED)

add_library(afml SHARED "")

target_sources(afml
  PRIVATE
  src/autograd/Variable.cpp
  src/autograd/Functions.cpp
  )

target_include_directories(afml
  PUBLIC
  ${ArrayFire_INCLUDE_DIRS}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  )

target_link_libraries(afml
  PUBLIC
  af
  )

set_target_properties(afml
  PROPERTIES
  VERSION "${ArrayFireML_VERSION}"
  SOVERSION "${ArrayFireML_VERSION_MAJOR}"
  CXX_STANDARD 11
  )


add_subdirectory(examples)
