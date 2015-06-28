#include "afml/afml.hpp"

using namespace afml;

// https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/googlenet_cudnn.lua
// Note that Lua index starts from 1.
// TODO: make_shared is very frequently used. Is MS a good alias macro for make_shared?
// #define MS make_shared
NodePtr inception(const int inputSize, std::initializer_list<int>& config) {
  shared_ptr < Concat > concat(new Concat(1));
  if (config[0][0] != 0) {
    shared_ptr < Sequential > conv1(new Sequential());
    conv1->add(
        make_shared < Convolution > (inputSize, config[0][0], 1, 1, 1, 1))->add(
        make_shared<ReLU>());
    concat->add(conv1);
  }

  shared_ptr < Sequential > conv3(new Sequential());
  conv3->add(make_shared < Convolution > (inputSize, config[1][0], 1, 1, 1, 1))
      ->add(make_shared<ReLU>());
  conv3->add(
      make_shared < Convolution
          > (config[1][0], config[1][1], 3, 3, 1, 1, 1, 1))->add(
      make_shared<ReLU>());
  concat->add(conv3);

  shared_ptr < Sequential > conv3xx(new Sequential());
  conv3xx->add(
      make_shared < Convolution > (inputSize, config[2][0], 1, 1, 1, 1))->add(
      make_shared<ReLU>());
  conv3xx->add(
      make_shared < Convolution
          > (config[2][0], config[2][1], 3, 3, 1, 1, 1, 1))->add(
      make_shared<ReLU>());
  conv3xx->add(
      make_shared < Convolution
          > (config[2][1], config[2][1], 3, 3, 1, 1, 1, 1))->add(
      make_shared<ReLU>());
  concat->add(conv3xx);

  shared_ptr < Sequential > pool(new Sequential());
  pool->add(make_shared < Padding > (3, 3, 1, 1));
  if (config[3][0] == PoolingType::MAX) {
    pool->add(make_shared < MaxPooling > (3, 3, 1, 1));
  } else if (config[3][0] == PoolingType::AVERAGE) {
    pool->add(make_shared < AveragePooling > (3, 3, 1, 1));
  } else {
    printf("Unknown pooling");
    exit(1);
  }

  if (config[3][1] != 0) {
    pool->add(
        make_shared < Convolution
            > (inputSize, config[3][1], 1, 1, 1, 1)->add(make_shared<ReLU>()));
  }
  concat->add(pool);
  return concat;

}

NodePtr createModel(int numGPU) {
  shared_ptr < Sequential > features(new Sequential());
  features->add(make_shared < Convolution > (3, 64, 7, 7, 2, 2, 3, 3))->add(
      make_shared<ReLU>());
  features->add(make_shared < MaxPooling > (3, 3, 2, 2));
  features->add(make_shared < Convolution > (64, 64, 1, 1))->add(
      make_shared<ReLU>());
  features->add(make_shared < Convolution > (64, 192, 3, 3, 1, 1, 1, 1))->add(
      make_shared<ReLU>());
  features->add(make_shared < MaxPooling > (3, 3, 2, 2));

  int uselessPlaceHoder = -1;
  features->add(inception(192, { { 64, uselessPlaceHoder }, { 64, 64 },
                              { 64, 96 }, { PoolingType::AVERAGE, 32 } }));
  features->add(inception(256, { { 64, uselessPlaceHoder }, { 64, 96 },
                              { 64, 96 }, { PoolingType::AVERAGE, 64 } }));
  features->add(inception(320, { { 0, uselessPlaceHoder }, { 128, 160 }, { 64,
      96 }, { PoolingType::MAX, 0 } }));
  features->add(make_shared < Convolution > (576, 576, 2, 2, 2, 2));
  features->add(inception(576, { { 224, uselessPlaceHoder }, { 64, 96 }, { 96,
      128 }, { PoolingType::AVERAGE, 128 } }));
  features->add(inception(576, { { 192, uselessPlaceHoder }, { 96, 128 }, { 96,
      128 }, { PoolingType::AVERAGE, 128 } }));
  features->add(inception(576, { { 160, uselessPlaceHoder }, { 128, 160 }, {
      128, 160 }, { PoolingType::AVERAGE, 96 } }));
  features->add(inception(576, { { 96, uselessPlaceHoder }, { 128, 192 }, { 160,
      192 }, { PoolingType::AVERAGE, 96 } }));

  shared_ptr < Sequential > mainBranch(new Sequential());
  mainBranch->add(inception(576, { { 0, uselessPlaceHoder }, { 128, 192 }, {
      192, 256 }, { PoolingType::MAX, 0 } }));
  mainBranch->add(make_shared < Convolution > (1024, 1024, 2, 2, 2, 2));
  mainBranch->add(inception(1024, { { 352 }, { 192, 320 }, { 160, 224 }, {
      PoolingType::AVERAGE, 128 } }));
  mainBranch->add(inception(1024, { { 352 }, { 192, 320 }, { 192, 224 }, {
      PoolingType::MAX, 128 } }));
  mainBranch->add(make_shared < AveragePooling > (7, 7, 1, 1));
  mainBranch->add(make_shared < View > (1024)->withNumInputDims(3));
  mainBranch->add(make_shared < Linear > (1024, 1000));
  mainBranch->add(make_shared<LogSoftmax>());

  shared_ptr < Sequential > auxClassifier(new Sequential());
  auxClassifier->add(make_shared < AveragePooling > (5, 5, 3, 3));
  auxClassifier->add(make_shared < Convolution > (576, 128, 1, 1, 1, 1));
  auxClassifier->add(make_shared < View > (128 * 4 * 4)->withNumInputDims(3));
  auxClassifier->add(make_shared < Linear > (128 * 4 * 4, 768));
  auxClassifier->add(make_shared<ReLU>());
  auxClassifier->add(make_shared < Linear > (768, 1000));
  auxClassifier->add(make_shared<LogSoftmax>());

  shared_ptr < Concat > splitter(new Concat(1));
  splitter->add(mainBranch)->add(auxClassifier);
  shared_ptr < Sequential > model = make_shared<Sequential>()->add(features)
      ->add(splitter);

  if (numGPU > 0) {
    shared_ptr < DataParallel > dp(new DataParallel(1));
    for (int i = 0; i < numGPU; ++i) {
      dp->add(i, root->clone());
    }
    return dp;
  }
  return model;
}

int main(int argc, char *argv[]) {
  int numGPU = atoi(argv[1]);
  NodePtr model = createModel(numGPU);
  printf(model->toString());
}
