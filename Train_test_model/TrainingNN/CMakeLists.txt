cmake_minimum_required(VERSION 3.0.0)
project(prj1 VERSION 0.1.0)
#

file(TO_CMAKE_PATH "$ENV{MXNET_ROOT}" MXNET_ROOT)

SET(MXNET_INCLUDE_SEARCH_PATHS
  ~/mxnet
  ~/mxnet/include
  ~/mxnet/include/mxnet
  ~/mxnet/cpp-package/include
  ~/mxnet/3rdparty/tvm/nnvm/include
  ${MXNET_ROOT}/include/mxnet
)

SET(MXNET_LIB_SEARCH_PATHS
~/mxnet/build/release
~/mxnet/src
~/mxnet/
~/mxnet/3rdparty/tvm/nnvm/include/
        /lib
        /lib/mxnet
        /lib64
        /usr/lib
        /usr/lib/mxnet
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/MXNET/lib

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
  ~/mxnet/3rdparty/tvm/nnvm/include
  ~/mxnet/3rdparty/dmlc-core/include
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



include_directories(${MxNet_INCLUDE_DIRS})
#
add_library( mp2
    SHARED
    networks_bn.cpp
    networks_bn.hpp

)

add_executable(mp1 main.cpp )

set_property(TARGET mp2 PROPERTY CXX_STANDARD 11)
set_target_properties(mp1
PROPERTIES
CXX_STANDARD 11
CXX_EXTENSIONS OFF
CXX_STANDARD_REQUIRED ON
POSITION_INDEPENDENT_CODE 1
)


target_link_libraries( mp1 mp2 )
target_link_libraries( mp1 ${MxNet_LIB} )


message(STATUS "MxNet lib: ${MxNet_LIB}" )
message(STATUS "C flags, Debug configuration: ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C flags, Release configuration: ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "C flags, Release configuration with Debug info: ${CMAKE_C_FLAGS_RELWITHDEBINFO}")
message(STATUS "C flags, minimal Release configuration: ${CMAKE_C_FLAGS_MINSIZEREL}")

message(STATUS "C++ flags, Debug configuration: ${CMAKE_CXX_FLAGS_DEBUG}")
message(STATUS "C++ flags, Release configuration: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "C++ flags, Release configuration with Debug info: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
message(STATUS "C++ flags, minimal Release configuration: ${CMAKE_CXX_FLAGS_MINSIZEREL}")





