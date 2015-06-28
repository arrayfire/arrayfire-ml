#!/bin/bash

NAME=afml

#include "afml/thrift/afml_constants.h"

thrift -r --gen cpp ${NAME}.thrift
sed -i "s|#include \"${NAME}_constants|#include \"${NAME}/thrift/${NAME}_constants|g" gen-cpp/*
sed -i "s|#include \"${NAME}_types|#include \"${NAME}/thrift/${NAME}_types|g" gen-cpp/*
if [ ! -d ../include/${NAME}/thrift ]; then
  mkdir ../include/${NAME}/thrift
fi
mv gen-cpp/*.h ../include/${NAME}/thrift
if [ ! -d ../src/${NAME}/thrift ]; then
  mkdir ../src/${NAME}/thrift
fi
mv gen-cpp/*.cpp ../src/${NAME}/thrift
rm -rf gen-cpp

# thrift -r --gen java afml.thrift
# thrift -r --gen js afml.thrift
# thrift -r --gen json afml.thrift
# thrift -r --gen lua afml.thrift
# thrift -r --gen py afml.thrift
