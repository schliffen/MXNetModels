# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.5)

# Set the project name
project (hello_headers)


# cpp version
set(CMAKE_CXX_STANDARD 14)


# Create a sources variable with a link to all cpp files to compile
set(SOURCES
    src/Hello.cpp
    src/main.cpp
)
#set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
#set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

include_directories("${PROJECT_SOURCE_DIR}")

include_directories(
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src
)
#

#file(GLOB all_SRCS
#        "${PROJECT_SOURCE_DIR}/include/*.h"
#		"${PROJECT_SOURCE_DIR}/include/*.hpp"
#        "${PROJECT_SOURCE_DIR}/src/*.cpp"
#		"${PROJECT_SOURCE_DIR}/src/*.c"
#        )

# Add an executable with the above sources
add_executable( ${PROJECT_NAME} ${SOURCES} )
#add_executable( hello_headers ${all_SRCS} )

# Set the directories that should be included in the build command for this target
# when running g++ these will be included as -I/directory/path/
 target_include_directories( ${PROJECT_NAME}
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include        
)

MESSAGE( ${SOURCES})

