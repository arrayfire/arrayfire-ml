function(build_example SRC)
  get_filename_component(src_name ${SRC} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRC})
  target_include_directories(${target}
    PRIVATE
    ${ArrayFire_INCLUDE_DIRS}
    ${ArrayFireML_INCLUDE_DIRS}
    )
  target_link_libraries(${target}
    PRIVATE
    af
    )
  target_compile_features(${target}
    PRIVATE cxx_range_for)
endfunction(build_example)

build_example(Activations.cpp)
build_example(FFNet.cpp)
build_example(Node.cpp)
build_example(perceptron.cpp)
build_example(Weights.cpp)
