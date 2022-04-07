cmake_minimum_required(VERSION 3.14)
project(TinyDnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

if (WIN32)
    include_directories("/usr/local/include/eigen3/")
elseif (UNIX)
    include_directories("/usr/local/include/eigen3/")
endif ()

add_executable(eigen main.cpp dnn.h dnn.cpp graph_manager.h graph_manager.cpp variable.h operator_bash.h)