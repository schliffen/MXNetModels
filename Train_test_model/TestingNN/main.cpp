/*
This was completed with pain! 2020
 */
#include <iostream>
#include <map>
#include <typeinfo>
#include <string>
#include <fstream>
#include <cstdlib>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"
#include "ModelHandle.hpp"
#include "ImageUtils.hpp"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace mxnet::cpp;
//
namespace fs = std::experimental::filesystem;
const cv::String keys =
    "{help h usage ? |      | print this message   }"
    "{@params        |<none>| path to trained parameters }"
    "{@image         |<none>| path to image }";

static Context global_ctx = Context::gpu();


int main(int argc, char**argv ) {
  //
  cv::CommandLineParser parser(argc, argv, keys);
  parser.about("CNN_Testing");
  if (parser.has("help") || argc == 1) {
    parser.printMessage();
    return 0;
  }
  /*Loading required ckasses*/

  Alexnetsymb   alexnet;
  model_handler load_mdl;
  //
  
  // introducing parameters
  //auto ctx = Context::gpu();
  // auto ctx = Context::cpu();

// #if MXNET_USE_CPU
  // ctx = Context::cpu();
// #endif
// check if image entered as argument
std::string saved_mdel_path;
std::string image_path;
  try{

    
   saved_mdel_path = parser.get<cv::String>(0);
   image_path = parser.get<cv::String>(1);
  //  params_path = 
  }
  catch(int e){
   saved_mdel_path = "../inception_bn_param_1";
   image_path = "/home/ali/PROJ_LAB/IMG_DATA/mnistasjpg/testSample/img_1.jpg"; //from MNIST dataset
  };

  std::cout<< "I entered to the program! " << std::endl;
// in order to use pretrained models .... TO BE DONE FOR LATER

// std::pair<std::map<std::string, NDArray>,
          // std::map<std::string, NDArray>> loaded_params 
          // = load_mdl.LoadNetParams(ctx, saved_mdel_path);

//-----------
// preparing test images 
//----------- Load params
std::map<std::string, NDArray> args_map;
std::map<std::string, NDArray> aux_map;
//
std::tie(args_map, aux_map) = load_mdl.LoadNetParams(global_ctx, saved_mdel_path);
//

//
// args_map["data"] = data;
// args_map["im_info"] = im_info;

NDArray::WaitAll();
// CheckMXnetError("Load parameters from file");

std::cout<< " Model is loaded successfully! " << std::endl;


//   auto Net = alexnet.netwrk(10);

//   /*args_map and aux_map is used for parameters' saving*/
//   std::map<std::string, NDArray> args_map;
//   std::map<std::string, NDArray> aux_map;


//   /*we should tell mxnet the shape of data and label*/
//   args_map["data"] = NDArray(Shape(1, 3, 256, 256), ctx);
//   args_map["label"] = NDArray(Shape(1), ctx);

//   cout<< typeid(args_map).name() << endl;
  

//   /*with data and label, executor can be generated automatically*/
//   auto *exec = Net.SimpleBind(ctx, args_map);
//   auto arg_names = Net.ListArguments();

  

//   aux_map = exec->aux_dict();
//   args_map = exec->arg_dict();


//   Accuracy acu_train, acu_val;
//   LogLoss logloss_val;




//   cout<< "Model is trained! parameters were saved!" << endl;
//   /*don't foget to release the executor*/
//   delete exec;
//   MXNotifyShutdown();
//   return 0;
}