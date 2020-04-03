//
//
//
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "networks_bn.hpp"
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;


Symbol NNet::ConvFactoryBN(Symbol data, int num_filter,
                     Shape kernel, Shape stride, Shape pad,
                     const std::string & name,
                     const std::string & suffix = "") {
  Symbol conv_w("conv_" + name + suffix + "_w"), conv_b("conv_" + name + suffix + "_b");

  Symbol conv = Convolution("conv_" + name + suffix, data,
                            conv_w, conv_b, kernel,
                            num_filter, stride, Shape(1, 1), pad);
  std::string name_suffix = name + suffix;
  Symbol gamma(name_suffix + "_gamma");
  Symbol beta(name_suffix + "_beta");
  Symbol mmean(name_suffix + "_mmean");
  Symbol mvar(name_suffix + "_mvar");
  Symbol bn = BatchNorm("bn_" + name + suffix, conv, gamma, beta, mmean, mvar);
  return Activation("relu_" + name + suffix, bn, "relu");
};

NNet nnt;

Symbol Inception::InceptionFactoryA(Symbol data, int num_1x1, int num_3x3red,
                         int num_3x3, int num_d3x3red, int num_d3x3,
                         PoolingPoolType pool, int proj,
                         const std::string & name) {
  Symbol c1x1 = nnt.ConvFactoryBN(data, num_1x1, Shape(1, 1), Shape(1, 1),
                              Shape(0, 0), name + "1x1");
  Symbol c3x3r = nnt.ConvFactoryBN(data, num_3x3red, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_3x3r");
  Symbol c3x3 = nnt.ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(1, 1),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = nnt.ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = nnt.ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = nnt.ConvFactoryBN(data = cd3x3, num_d3x3, Shape(3, 3), Shape(1, 1),
                        Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling(name + "_pool", data,
                           Shape(3, 3), pool, false, false,
                           PoolingPoolingConvention::kValid,
                           Shape(1, 1), Shape(1, 1));
  Symbol cproj = nnt.ConvFactoryBN(pooling, proj, Shape(1, 1), Shape(1, 1),
                               Shape(0, 0), name + "_proj");
  std::vector<Symbol> lst;
  lst.push_back(c1x1);
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(cproj);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
};


Symbol Inception::InceptionFactoryB(Symbol data, int num_3x3red, int num_3x3,
                         int num_d3x3red, int num_d3x3, const std::string & name) {
  Symbol c3x3r = nnt.ConvFactoryBN(data, num_3x3red, Shape(1, 1),
                               Shape(1, 1), Shape(0, 0),
                               name + "_3x3", "_reduce");
  Symbol c3x3 = nnt.ConvFactoryBN(c3x3r, num_3x3, Shape(3, 3), Shape(2, 2),
                              Shape(1, 1), name + "_3x3");
  Symbol cd3x3r = nnt.ConvFactoryBN(data, num_d3x3red, Shape(1, 1), Shape(1, 1),
                                Shape(0, 0), name + "_double_3x3", "_reduce");
  Symbol cd3x3 = nnt.ConvFactoryBN(cd3x3r, num_d3x3, Shape(3, 3), Shape(1, 1),
                               Shape(1, 1), name + "_double_3x3_0");
  cd3x3 = nnt.ConvFactoryBN(cd3x3, num_d3x3, Shape(3, 3), Shape(2, 2),
                        Shape(1, 1), name + "_double_3x3_1");
  Symbol pooling = Pooling("max_pool_" + name + "_pool", data,
                           Shape(3, 3), PoolingPoolType::kMax,
                           false, false, PoolingPoolingConvention::kValid,
                           Shape(2, 2), Shape(1, 1));
  std::vector<Symbol> lst;
  lst.push_back(c3x3);
  lst.push_back(cd3x3);
  lst.push_back(pooling);
  return Concat("ch_concat_" + name + "_chconcat", lst, lst.size());
};

Symbol Inception::InceptionSymbol(int num_classes) {
  // data and label
  Symbol data = Symbol::Variable("data");
  Symbol data_label = Symbol::Variable("data_label");

  // stage 1
  Symbol conv1 = nnt.ConvFactoryBN(data, 64, Shape(7, 7), Shape(2, 2), Shape(3, 3), "conv1");
  Symbol pool1 = Pooling("pool1", conv1, Shape(3, 3), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  // stage 2
  Symbol conv2red = nnt.ConvFactoryBN(pool1, 64, Shape(1, 1), Shape(1, 1),  Shape(0, 0), "conv2red");
  Symbol conv2 = nnt.ConvFactoryBN(conv2red, 192, Shape(3, 3), Shape(1, 1), Shape(1, 1), "conv2");
  Symbol pool2 = Pooling("pool2", conv2, Shape(3, 3), PoolingPoolType::kMax,
      false, false, PoolingPoolingConvention::kValid, Shape(2, 2));

  // stage 3
  Symbol in3a = InceptionFactoryA(pool2, 64, 64, 64, 64, 96, PoolingPoolType::kAvg, 32, "3a");
  Symbol in3b = InceptionFactoryA(in3a, 64, 64, 96, 64, 96, PoolingPoolType::kAvg, 64, "3b");
  Symbol in3c = InceptionFactoryB(in3b, 128, 160, 64, 96, "3c");

  // stage 4
  Symbol in4a = InceptionFactoryA(in3c, 224, 64, 96, 96, 128, PoolingPoolType::kAvg, 128, "4a");
  Symbol in4b = InceptionFactoryA(in4a, 192, 96, 128, 96, 128,  PoolingPoolType::kAvg, 128, "4b");
  Symbol in4c = InceptionFactoryA(in4b, 160, 128, 160, 128, 160, PoolingPoolType::kAvg, 128, "4c");
  Symbol in4d = InceptionFactoryA(in4c, 96, 128, 192, 160, 192,  PoolingPoolType::kAvg, 128, "4d");
  Symbol in4e = InceptionFactoryB(in4d, 128, 192, 192, 256, "4e");

  // stage 5
  Symbol in5a = InceptionFactoryA(in4e, 352, 192, 320, 160, 224, PoolingPoolType::kAvg, 128, "5a");
  Symbol in5b = InceptionFactoryA(in5a, 352, 192, 320, 192, 224, PoolingPoolType::kMax, 128, "5b");

  // average pooling
  Symbol avg = Pooling("global_pool", in5b, Shape(7, 7), PoolingPoolType::kAvg);

  // classifier
  Symbol flatten = Flatten("flatten", avg);
  Symbol conv1_w("conv1_w"), conv1_b("conv1_b");
  Symbol fc1 = FullyConnected("fc1", flatten, conv1_w, conv1_b, num_classes);
  return SoftmaxOutput("softmax", fc1, data_label);
};

// void model_handler::save_model(int save_args, std::string prefix, std::string data_name, std::string label_name ) {

//     /*we do not want to save the data and label*/
//     //save_args.erase(save_args.find( data_name ));
//     //save_args.erase(save_args.find( label_name) );

//     /*the alexnet does not get any aux array, so we do not need to save
//      * aux_map*/
//     //for (auto iter : exec->aux_dict()) {
//     //  save_args.insert({"aux:" + iter.first, iter.second});};
    
//     //NDArray::Save(prefix, save_args);
//     //
//     std::cout << "The model is saved ... "<< std::endl;
//     };

std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>>
model_handler::LoadNetParams(const mxnet::cpp::Context& ctx, const std::string& param_file) {
  using namespace mxnet::cpp;
  //---------- Load parameters
  std::map<std::string, NDArray> paramters;
  NDArray::Load(param_file, nullptr, &paramters);
  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;
  for (const auto& k : paramters) {
    if (k.first.substr(0, 4) == "aux:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      aux_map[name] = k.second.Copy(ctx);
      aux_map[name].WaitAll();
    }
    if (k.first.substr(0, 4) == "arg:") {
      auto name = k.first.substr(4, k.first.size() - 4);
      args_map[name] = k.second.Copy(ctx);
      args_map[name].WaitAll();
    }
  }
  return std::make_pair(args_map, aux_map);
}

void model_handler::SaveNetParamsFromExec(const std::string& param_file, mxnet::cpp::Executor* exec) {
  NDArray::WaitAll();
  std::map<std::string, NDArray> params;
  for (auto& iter : exec->arg_dict()) {
    if (iter.first.rfind("data", 0) != 0 &&
        iter.first.rfind("im_info", 0) != 0 &&
        iter.first.rfind("gt_boxes", 0) != 0 &&
        iter.first.rfind("label", 0) != 0 &&
        iter.first.rfind("bbox_target", 0) != 0 &&
        iter.first.rfind("bbox_weight", 0) != 0)
      params.insert({"arg:" + iter.first, iter.second});
  }
  for (auto iter : exec->aux_dict()) {
    params.insert({"aux:" + iter.first, iter.second});
  }
  NDArray::Save(param_file, params);
};

void model_handler::saveNetParams(std::pair<std::map<std::string, NDArray>,
          std::map<std::string, NDArray>> save_args, const std::string& prefix, const std::string& data_name,   const std::string& label_name){

    //std::cout<< "cleaning the model 2" <<std::endl; 
    std::cout << "saving the model ... "<< std::endl;
    // NDArray::Save(prefix, save_args);
}