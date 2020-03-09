cmake_minimum_required (VERSION 3.1)

# including required directories
include_directories( ../include/)


add_executable(exam1.out main.cpp )

# linking them toghether
target_link_libraries( exam1.out Foo )
