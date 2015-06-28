FIND_PACKAGE(ArrayFire)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
...

ADD_EXECUTABLE(some_executable ...)
TARGET_LINK_LIBRARIES(some_executable ${ArrayFire_LIBRARIES} )