cmake_minimum_required(VERSION 3.14)
project(TinyDnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Debug")

if (MINGW)
    include_directories("/usr/local/include/eigen3/")
elseif (UNIX)
    include_directories("/usr/local/include/eigen3/")
endif ()

add_executable(eigen main.cpp
        variable.h variable.cpp
        full_connect.h full_connect.cpp
        operator_base.cpp operator_base.h
        graph_manager.h
        cross_entropy_loss.cpp cross_entropy_loss.h)
