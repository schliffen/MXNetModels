#include "mxnet-cpp/MxNetCpp.h"
#include "utils.h"

class Alexnetsymb {
public:
Symbol netwrk(int num_classes);

};
//

class model_handler{
    public:
      std::pair<std::map<std::string, NDArray>,
          std::map<std::string, NDArray>> LoadNetParams( const Context& , const std::string& );
      void  SaveNetParamsFromExec( const std::string& , Executor*  );
    
      void  saveNetParams(std::pair<std::map<std::string, mxnet::cpp::NDArray>,
          std::map<std::string, mxnet::cpp::NDArray>>, const std::string&, const std::string&,   const std::string& ); 
};
