cmake_minimum_required(VERSION 3.0.0)
project(prj1 VERSION 0.1.0)
# #
# file(TO_CMAKE_PATH "$ENV{MXNET_ROOT}" MXNET_ROOT)

# #
# #set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/)
# #find_package(MxNet REQUIRED)
# #include_directories(${MxNet_INCLUDE_DIRS})
# #

# add_executable(mp1 main.cpp )

# target_link_libraries( mp1 mp2 )
# target_link_libraries( mp1 ${MxNet_LIB} )

set(CMAKE_PREFIX_PATH /home/ali/installation/opencv-4.2.0/lib/cmake/opencv4)
set(OPENCV_INCLUDE_DIR /home/ali/installation/opencv3/include)
set(OPENCV_LIB_DIR /home/ali/installation/opencv3/lib)

find_package( OpenCV REQUIRED )

file(TO_CMAKE_PATH "$ENV{MXNET_ROOT}" MXNET_ROOT)

SET(MXNET_INCLUDE_SEARCH_PATHS
  ~/BMXNet-v2
  ~/BMXNet-v2/include/mxnet
  ~/home/ali/BMXNet-v2/cpp-package/include/mxnet-cpp
  /home/ali/BMXNet-v2/3rdparty/tvm/nnvm/include
  /usr/include/mxnet
  /usr/local/include/mxnet
  /opt/mxnet/include/mxnet
  /opt/staging/incubator-mxnet/include/mxnet
  /usr/local/opt/mxnet/include/mxnet
  ${MXNET_ROOT}/include/mxnet
)

SET(MXNET_LIB_SEARCH_PATHS
/home/ali/BMXNet-v2/build
/home/ali/BMXNet-v2/src
/home/ali/BMXNet-v2
/home/ali/BMXNet-v2/3rdparty/tvm/nnvm/include/
        /lib
        /lib/mxnet
        /lib64
        /usr/lib
        /usr/lib/mxnet
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/MXNET/lib
	/opt/stating/
	/opt/staging/incubator-mxnet/lib
        /usr/local/opt/openblas/lib
#        ${PROJECT_SOURCE_DIR}/3rdparty/mxnet/lib
#        ${PROJECT_SOURCE_DIR}/thirdparty/mxnet/lib
#        ${MXNET_ROOT}
#        ${MXNET_ROOT}/lib
 )


FIND_PATH(MxNet_INCLUDE_DIR NAMES c_api.h PATHS ${MXNET_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(MxNet_LIB NAMES mxnet PATHS ${MXNET_LIB_SEARCH_PATHS})

SET(MxNet_FOUND ON)

#    Check include files
IF(NOT MxNet_INCLUDE_DIR)
    SET(MxNet_FOUND OFF)
    MESSAGE(STATUS "Could not find MxNet include. Turning MxNet_FOUND off")
ENDIF()

message( "include mxnet dir :" ${MxNet_INCLUDE_DIR})

# Set all the include paths relative to the one found
set(MxNet_INCLUDE_DIRS 
  ${MxNet_INCLUDE_DIR}/../
  ${MxNet_INCLUDE_DIR}/../../cpp-package/include/
  ${MxNet_INCLUDE_DIR}/../../dmlc-core/include/
  ${MxNet_INCLUDE_DIR}/../../mshadow
  ${MxNet_INCLUDE_DIR}/../../nnvm/include
  ${MxNet_INCLUDE_DIR}/dlpack/include
  /home/ali/BMXNet-v2/3rdparty/tvm/nnvm/include
  /home/ali/BMXNet-v2/3rdparty/dmlc-core/include
)

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
# opencv Include
include_directories(${OPENCV_INCLUDE_DIR})


#message( ${MxNet_LIB} )

include_directories(${MxNet_INCLUDE_DIRS})
#
add_library( mp2
    SHARED
    stest.cpp
    stest.hpp
)


add_executable(mp1 main.cpp )

set_property(TARGET mp1 PROPERTY CXX_STANDARD 11)

target_link_libraries( mp1 mp2 )
target_link_libraries( mp1 ${MxNet_LIB} ${OpenCV_LIBS})