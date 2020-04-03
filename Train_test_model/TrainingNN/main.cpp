//
#include <iostream>
#include <map>
#include <typeinfo>
#include <string>
#include <fstream>
#include <vector>
#include "networks_bn.hpp"
#include "utils.h"
//#include "ndarray.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

int main(int argc, char const *argv[]) {
  int batch_size = 40;
  int max_epoch = argc > 1 ? strtol(argv[1], NULL, 10) : 100;
  float learning_rate = 1e-2;
  float weight_decay = 1e-4;
  //
  const char * data_name = "data";
  const char * label_name = "data_label";

  std::cout << "Max epochs: " << max_epoch << std::endl;
  auto ctx = Context::gpu();
#if MXNET_USE_CPU
  ctx = Context::cpu();
#endif

    // defining networks
 
    NNet nn;
    Inception incpt;
    model_handler load_mdl;

  // in order to use pretrained models .... TO BE DONE FOR LATER
  std::string saved_mdel = "../inception_bn_param_1";
  std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>> loaded_model = load_mdl.LoadNetParams(ctx, saved_mdel);
  std::cout<<"Pretrained Model is loaded successfully! "<< std::endl;

  auto inception_bn_net = incpt.InceptionSymbol(10);

  std::map<std::string, NDArray> args_map;
  std::map<std::string, NDArray> aux_map;

  args_map[data_name] = NDArray(Shape(batch_size, 3, 224, 224), ctx);

  args_map[label_name] = NDArray(Shape(batch_size), ctx);

  inception_bn_net.InferArgsMap(ctx, &args_map, args_map);


  std::vector<std::string> data_files = { "../../data/mnist_data/train-images-idx3-ubyte",
                                          "../../data/mnist_data/train-labels-idx1-ubyte",
                                          "../../data/mnist_data/t10k-images-idx3-ubyte",
                                          "../../data/mnist_data/t10k-labels-idx1-ubyte"
                                        };

  auto train_iter =  MXDataIter("MNISTIter");
  setDataIter(&train_iter, "Train", data_files, batch_size);
  auto val_iter = MXDataIter("MNISTIter");
  setDataIter(&val_iter, "Label", data_files, batch_size);

  // initialize parameters
  Xavier xavier = Xavier(Xavier::gaussian, Xavier::in, 2);
  for (auto &arg : args_map) {
    xavier(arg.first, &arg.second);
  }

  Optimizer* opt = OptimizerRegistry::Find("adam"); //"sgd");
  opt->SetParam("momentum", 0.9)
     ->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("clip_gradient", 10)
     ->SetParam("lr", learning_rate)
     ->SetParam("wd", weight_decay);

  auto *exec = inception_bn_net.SimpleBind(ctx, args_map);
  auto arg_names = inception_bn_net.ListArguments();

  //std::cout<< "args_map type: "<< typeid(args_map).name() << std::endl;

  // Create metrics
  Accuracy train_acc, val_acc;

  for (int iter = 0; iter < max_epoch; ++iter) {
    LG << "Epoch: " << iter;
    train_iter.Reset();
    train_acc.Reset();
    int subiter =1;
    while (train_iter.Next()) {
      auto data_batch = train_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();
      // Update parameters
      for (size_t i = 0; i < arg_names.size(); ++i) {
        if (arg_names[i] == data_name || arg_names[i] == label_name ) continue;
        opt->Update(i, exec->arg_arrays[i], exec->grad_arrays[i]);
      }

     NDArray::WaitAll();
     train_acc.Update(data_batch.label, exec->outputs[0]);
     std::cout<< "Completion %: "<<  subiter * batch_size/600 << std::endl;
     subiter++;
    
    }

    val_iter.Reset();
    val_acc.Reset();
 
    while (val_iter.Next()) {
      auto data_batch = val_iter.GetDataBatch();
      data_batch.data.CopyTo(&args_map["data"]);
      data_batch.label.CopyTo(&args_map["data_label"]);
      NDArray::WaitAll();
      exec->Forward(false);
      NDArray::WaitAll();
      val_acc.Update(data_batch.label, exec->outputs[0]);
    }
    LG << "Train Accuracy: " << train_acc.Get();
    LG << "Validation Accuracy: " << val_acc.Get();

    /*save the parameters*/
    // naming conventions
    std::stringstream ss;
    ss << iter;
    std::string iter_str;
    ss >> iter_str;
    //
    std::string prefix = "inception_bn_param_" + iter_str; 
    //
    std::cout<< "args_map type: "<< typeid(args_map).name() << std::endl;

    /*the alexnet does not get any aux array, so we do not need to save
     * aux_map*/
    auto save_args = args_map;
        /*we do not want to save the data and label*/   
    save_args.erase(save_args.find(data_name));
    save_args.erase(save_args.find(label_name));
    for (auto iter : exec->aux_dict()) {
     save_args.insert({"aux:" + iter.first, iter.second});};
    //
    NDArray::Save(prefix, save_args);
    // mdlsvr.SaveNetParams( save_args, prefix, data_name, label_name);

    LG << "ITER: " << iter << " Saving to..." << prefix;
    

    /*save the parameters*/
    /*
    std::stringstream ss;
    ss << iter;
    std::string iter_str;
    ss >> iter_str;
    std::string save_path_param = "inception_bn_param_" + iter_str;
    auto save_args = args_map;
    // we do not want to save the data and label
    save_args.erase(save_args.find("data"));
    save_args.erase(save_args.find("label"));
    //the alexnet does not get any aux array, so we do not need to save aux_map
    LG << "ITER: " << iter << " Saving to..." << save_path_param;
    */
    /*
      This is to insert aux params!
  }
    */
  
    //for (auto iter : exec->aux_dict()) {
    //save_args.insert({"aux:" + iter.first, iter.second}); };
    //NDArray::Save(save_path_param, save_args);

  }
    
  std::cout<< "Model is trained! parameters were saved!" << std::endl;

  // freeing the memory
  delete exec;

  MXNotifyShutdown();

  //save and load "exec" to make predictions
  std::cout<< "type of lr: "<< typeid(learning_rate).name() << std::endl;

  return 0;
}
