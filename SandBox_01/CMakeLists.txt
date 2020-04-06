cmake_minimum_required(VERSION 3.6.0)
project(mxnet_p1 VERSION 0.1.0)
#

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_FLAGS "-std=c++11" ) # -fopenmp -pthread") # -msse3 -Wall -Wextra -Wno-gnu-include-next -Wno-int-in-bool-context -Wno-unused-parameter -Wno-unknown-pragmas -Wno-sign-compare")
# 

set(CUDA_NVCC_FLAGS "-std=c++11;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wall;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wextra;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unused-parameter;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-unknown-pragmas;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-int-in-bool-context;")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options;-Wno-sign-compare;")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr;")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

set(CUDA_NVCC_FLAGS_RELEASE "-O3;-DNDEBUG;")
set(CUDA_NVCC_FLAGS_DEBUG "-g;-O0;")
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

set(OPENCV_INCLUDE_DIR /home/ali/installation/opencv3/include)
set(OPENCV_LIB_DIR /home/ali/installation/opencv3/lib)
# set(LLVM_INCLUDE_DIRS /home/ali/installation/llvm-8/include)
# set(LLVM_LIBS /home/ali/installation/llvm-8/lib)
# set(LLVM_DIR /home/ali/installation/llvm-8/lib/cmake/llvm)
#set(OpenMP_DIR )
#
file(TO_CMAKE_PATH "$ENV{MXNET_ROOT}" MXNET_ROOT)

# looking for opencv library
set(CMAKE_PREFIX_PATH /home/ali/installation/opencv3/share/OpenCV)
find_package( OpenCV REQUIRED )
find_package( BLAS REQUIRED )
find_package( CUDA REQUIRED )
# find_package( OpenMP REQUIRED ) 
# find_package(LLVM REQUIRED 
# /home/ali/installation/llvm-8/include
# /home/ali/installation/llvm-8/lib
# /home/ali/installation/llvm-8
# /home/ali/installation/llvm-8/lib/cmake/llvm
# NO_DEFAULT_PATH)

#/home/ali/installation/llvm-8/include/clang
#/home/ali/installation/llvm-8/include
# NO_DEFAULT_PATH)

# looking for mxnet library -----
set(MXNET_includes /home/ali/BMXNet-v2)
SET(MXNET_INCLUDE_SEARCH_PATHS
${MXNET_includes}/include/mxnet
${MXNET_includes}/cpp-package/include
${MXNET_includes}/3rdparty/dmlc-core/include/
${MXNET_includes}/3rdparty/mshadow
${MXNET_includes}/3rdparty/tvm/include
${MXNET_includes}/3rdparty/dlpack/include
${MXNET_includes}/3rdparty/mkldnn/include
${MXNET_includes}/3rdparty/tvm/nnvm/include/nnvm
)

SET(MXNET_LIB_SEARCH_PATHS
${MXNET_includes}/build
${MXNET_includes}/build/3rdparty/dmlc-core
${MXNET_includes}/build/lib
${MXNET_includes}/build/tools/binary_converter
 )
# #
 # Find path and lib for mxnet 
 # Set all the include paths relative to the one found

# set(MxNet_INCLUDE_DIRS 
# ${MXNET_ROOT}/include
# ${MXNET_ROOT}/cpp-package/include
# ${MXNET_ROOT}/3rdparty/dmlc-core/include/
# ${MXNET_ROOT}/3rdparty/mshadow
# ${MXNET_ROOT}/3rdparty/tvm/include
# ${MXNET_ROOT}/3rdparty/dlpack/include
# ${MXNET_ROOT}/3rdparty/mkldnn/include
# ${MXNET_ROOT}/3rdparty/tvm/nnvm/include
# )
# set(MxNet_LIB
# ${MXNET_ROOT}/build
# ${MXNET_ROOT}/build/3rdparty/dmlc-core
# ${MXNET_ROOT}/build/lib
# ${MXNET_ROOT}/build/tools/binary_converter
# # ${MXNET_ROOT}/
# ${MXNET_ROOT}/
# # ${MXNET_ROOT}/
# # ${MXNET_ROOT}/
# # ${MXNET_ROOT}/
# )

FIND_PATH(MxNet_INCLUDE_DIR NAMES c_api.h PATHS ${MXNET_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(MxNet_LIB NAMES mxnet PATHS ${MXNET_LIB_SEARCH_PATHS})

SET(MxNet_FOUND ON)
# Find path and lib for opencv

#    Check include files
IF(NOT MxNet_INCLUDE_DIR)
    SET(MxNet_FOUND OFF)
    MESSAGE(STATUS "Could not find MxNet include. Turning MxNet_FOUND off")
ENDIF()

message( "include mxnet dir :" ${MxNet_INCLUDE_DIR})



#    Check libraries
IF(NOT MxNet_LIB)
    SET(MxNet_FOUND OFF)
    MESSAGE(STATUS "Could not find MxNet lib. Turning MxNet_FOUND off")
ENDIF()

IF (MxNet_FOUND)
  IF (NOT MxNet_FIND_QUIETLY)
    MESSAGE(STATUS "Found MxNet libraries: ${MxNet_LIB}")
    MESSAGE(STATUS "Found MxNet include: ${MxNet_INCLUDE_DIRS}")
  ENDIF (NOT MxNet_FIND_QUIETLY)
ELSE (MxNet_FOUND)
  IF (MxNet_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find MxNet")
  ENDIF (MxNet_FIND_REQUIRED)
ENDIF (MxNet_FOUND)

MARK_AS_ADVANCED(
    MxNet_INCLUDE_DIRS
    MxNet_LIB
)

set(MxNet_INCLUDE_DIRS 
  ${MxNet_INCLUDE_DIR}/../
  ${MxNet_INCLUDE_DIR}/../../cpp-package/include/
  ${MxNet_INCLUDE_DIR}/../../3rdparty/dmlc-core/include/
  ${MxNet_INCLUDE_DIR}/../../3rdparty/mshadow
  ${MxNet_INCLUDE_DIR}/../../3rdparty/tvm/include
  ${MxNet_INCLUDE_DIR}/../../tvm/nnvm/include
  ${MxNet_INCLUDE_DIR}/../../3rdparty/dlpack/include
  ${MXNET_includes}/3rdparty/tvm/nnvm/include/
)

# if (OPENMP_FOUND)
#     message( STATUS "OPENMP FOUND! ")
#     #include_directories("${OPENMP_INCLUDES}")
#     link_directories("${OPENMP_LIBRARIES}")
#     set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    
    # set(CMAKE_C_COMPILER clang)
    # set(CMAKE_CXX_COMPILER clang++)
    # if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
    #     set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    #     set(OpenMP_C_LIB_NAMES "omp")
    #     set(OpenMP_omp_LIBRARY omp)
    # endif()

    # if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
    #     set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    #     set(OpenMP_CXX_LIB_NAMES "omp")
    #     set(OpenMP_omp_LIBRARY omp)
    # endif()
# endif(OPENMP_FOUND)

# MXNET Include
include_directories(${MxNet_INCLUDE_DIRS})

# opencv Include
include_directories(${OPENCV_INCLUDE_DIR})

# llvm and openmp
# include_directories(${LLVM_INCLUDE_DIRS})

# cuda include dir
include_directories(${CUDA_INCLUDE_DIRS})

#/home/ali/installation/opencv3
link_directories(${OPENCV_LIB_DIR})
link_directories(${MxNet_LIB})
# link_directories(${LLVM_LIBS})
#

#
add_library( mp2
    SHARED
    ./includes/mxlib_01.cpp
    ./includes/mxlib_01.hpp

)
include_directories(./includes )
# llvm_map_components_to_libnames(llvm_libs support core irreader )

# set(CMAKE_C_COMPILER   /home/ali/installation/llvm-8/bin/clang)
# set(CMAKE_CXX_COMPILER /home/ali/installation/llvm-8/bin/clang++)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")




cuda_add_executable(mxnet_p1 ./src/main.cpp)

# add_executable(mxnet_p1 main.cpp )

set_property(TARGET mp2 PROPERTY CXX_STANDARD 11)
set_target_properties(mxnet_p1
PROPERTIES
CXX_STANDARD 11
CXX_EXTENSIONS OFF
CXX_STANDARD_REQUIRED ON
POSITION_INDEPENDENT_CODE 1
)

target_link_libraries( mxnet_p1 mp2 )
target_link_libraries( mxnet_p1 ${MxNet_LIB} ${BLAS_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES})


message(STATUS "MxNet lib: ${MxNet_LIB}" )
message(STATUS "C flags, Debug configuration: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C flags, Release configuration: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "C flags, Release configuration with Debug info: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message(STATUS "C flags, minimal Release configuration: ${CMAKE_C_FLAGS_MINSIZEREL}")

message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")




