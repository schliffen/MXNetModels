cmake_minimum_required (VERSION 3.1)

# including required directories
include_directories( ../include/)

set(CMAKE_BUILD_TYPE Debug)

add_executable(exam2.out main.cpp )

# linking them toghether
target_link_libraries( exam2.out Foo )
