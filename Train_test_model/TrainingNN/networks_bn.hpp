
//
//
//
#include <map>
#include <string>
#include <fstream>
#include <vector>
#include "utils.h"
#include "mxnet-cpp/MxNetCpp.h"

using namespace mxnet::cpp;

class NNet{ 
  public:
  Symbol ConvFactoryBN(Symbol , int,
                     Shape, Shape, Shape,
                      const std::string &,
                      const std::string &); 
};


class Inception{
  public:
  Symbol InceptionFactoryA(Symbol , int , int ,
                         int , int , int ,
                         PoolingPoolType , int ,
                         const std::string & ); 

  Symbol InceptionFactoryB(Symbol , int , int ,
                         int , int , const std::string & );
  Symbol InceptionSymbol(int );                       
};


class model_handler{
    public:
      std::pair<std::map<std::string, NDArray>,
          std::map<std::string, NDArray>> LoadNetParams( const Context& , const std::string& );
      void  SaveNetParamsFromExec( const std::string& , Executor*  );
    
      void  saveNetParams(std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>>, const std::string&, const std::string&,   const std::string& ); 
};