cmake_minimum_required(VERSION 3.3)
project(Kmedians)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp Kmedians.cpp Kmedians.h ArrayUtils.h)
add_executable(Kmedians ${SOURCE_FILES})