cmake_minimum_required(VERSION 3.3)
project(KmediansBinary)
if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-O3")        ## Optimize
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fpermissive")
find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cc imgHelpers.cc imgHelpers.hpp precomp.hpp CONSTANTS.h latch.cpp global_descriptors.cc global_descriptors.hpp
        bag_of_words.cc bag_of_words.hpp ArrayUtils.h KmediansBinary.cpp KmediansBinary.h)
add_executable(feature_extractor ${SOURCE_FILES})
target_link_libraries( feature_extractor ${OpenCV_LIBS} )