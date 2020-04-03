
#include "ModelHandle.hpp"
#include "mxnet-cpp/MxNetCpp.h"

using namespace std;
using namespace mxnet::cpp;


Symbol Alexnetsymb::netwrk(int num_classes) {
  auto input_data = Symbol::Variable("data");
  auto target_label = Symbol::Variable("label");
  /*stage 1*/
  auto conv1 = Operator("Convolution")
                   .SetParam("kernel", Shape(11, 11))
                   .SetParam("num_filter", 96)
                   .SetParam("stride", Shape(4, 4))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(0, 0))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", input_data)
                   .CreateSymbol("conv1");
  auto relu1 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv1)
                   .CreateSymbol("relu1");
  auto pool1 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu1)
                   .CreateSymbol("pool1");
  auto lrn1 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool1)
                  .CreateSymbol("lrn1");
  /*stage 2*/
  auto conv2 = Operator("Convolution")
                   .SetParam("kernel", Shape(5, 5))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(2, 2))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn1)
                   .CreateSymbol("conv2");
  auto relu2 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv2)
                   .CreateSymbol("relu2");
  auto pool2 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max") /*avg,max,sum */
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu2)
                   .CreateSymbol("pool2");
  auto lrn2 = Operator("LRN")
                  .SetParam("nsize", 5)
                  .SetParam("alpha", 0.0001)
                  .SetParam("beta", 0.75)
                  .SetParam("knorm", 1)
                  .SetInput("data", pool2)
                  .CreateSymbol("lrn2");
  /*stage 3*/
  auto conv3 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", lrn2)
                   .CreateSymbol("conv3");
  auto relu3 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv3)
                   .CreateSymbol("relu3");
  auto conv4 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 384)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu3)
                   .CreateSymbol("conv4");
  auto relu4 = Operator("Activation")
                   .SetParam("act_type", "relu") /*relu,sigmoid,softrelu,tanh */
                   .SetInput("data", conv4)
                   .CreateSymbol("relu4");
  auto conv5 = Operator("Convolution")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("num_filter", 256)
                   .SetParam("stride", Shape(1, 1))
                   .SetParam("dilate", Shape(1, 1))
                   .SetParam("pad", Shape(1, 1))
                   .SetParam("num_group", 1)
                   .SetParam("workspace", 512)
                   .SetParam("no_bias", false)
                   .SetInput("data", relu4)
                   .CreateSymbol("conv5");
  auto relu5 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", conv5)
                   .CreateSymbol("relu5");
  auto pool3 = Operator("Pooling")
                   .SetParam("kernel", Shape(3, 3))
                   .SetParam("pool_type", "max")
                   .SetParam("global_pool", false)
                   .SetParam("stride", Shape(2, 2))
                   .SetParam("pad", Shape(0, 0))
                   .SetInput("data", relu5)
                   .CreateSymbol("pool3");
  /*stage4*/
  auto flatten =
      Operator("Flatten").SetInput("data", pool3).CreateSymbol("flatten");
  auto fc1 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", flatten)
                 .CreateSymbol("fc1");
  auto relu6 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc1)
                   .CreateSymbol("relu6");
  auto dropout1 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu6)
                      .CreateSymbol("dropout1");
  /*stage5*/
  auto fc2 = Operator("FullyConnected")
                 .SetParam("num_hidden", 4096)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout1)
                 .CreateSymbol("fc2");
  auto relu7 = Operator("Activation")
                   .SetParam("act_type", "relu")
                   .SetInput("data", fc2)
                   .CreateSymbol("relu7");
  auto dropout2 = Operator("Dropout")
                      .SetParam("p", 0.5)
                      .SetInput("data", relu7)
                      .CreateSymbol("dropout2");
  /*stage6*/
  auto fc3 = Operator("FullyConnected")
                 .SetParam("num_hidden", num_classes)
                 .SetParam("no_bias", false)
                 .SetInput("data", dropout2)
                 .CreateSymbol("fc3");
  auto softmax = Operator("SoftmaxOutput")
                     .SetParam("grad_scale", 1)
                     .SetParam("ignore_label", -1)
                     .SetParam("multi_output", false)
                     .SetParam("use_ignore", false)
                     .SetParam("normalization", "null") /*batch,null,valid */
                     .SetInput("data", fc3)
                     .SetInput("label", target_label)
                     .CreateSymbol("softmax");
  return softmax;
};

//
// code for load and running the model 
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
