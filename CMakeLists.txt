cmake_minimum_required(VERSION 3.9)

project( stereo_estimation )

###############
# packages #
###############

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(OpenMP)

###############
# executables #
###############

add_executable( stereo_matching src/main.cpp )
target_link_libraries( stereo_matching ${OpenCV_LIBS} )

if(OpenMP_CXX_FOUND)
    target_link_libraries(stereo_matching OpenMP::OpenMP_CXX)
endif()