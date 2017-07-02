cmake_minimum_required(VERSION 3.5.2)

project(ArrayFireML
  VERSION 0.1.0
  LANGUAGES C CXX)

find_package(ArrayFire REQUIRED)
set(ArrayFireML_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_subdirectory(examples)
