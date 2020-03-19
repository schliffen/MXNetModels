cmake_minimum_required (VERSION 3.1)

# Setup projectname
project (HelloProject)

#//Choose the type of build, options are: None Debug Release RelWithDebInfo
#// MinSizeRel ...
#CMAKE_BUILD_TYPE:STRING=Debug

# Compile and link main.cpp and foo.cpp
# into the executable Hello
add_subdirectory( src )
add_subdirectory( include )   
set(CMAKE_BUILD_TYPE Debug)
#add_executable(exam1.out src/main.cpp src/foo.cpp)

#SET( mesa "This is root dir ")
#MESSAGE( $ {mesa})
