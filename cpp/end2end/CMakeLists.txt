# cmake_minimum_required(VERSION 2.6)

# project(yolo)

# add_definitions(-std=c++11)

# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)
# # add_definitions("-Wall -g")
# find_package(CUDA REQUIRED)
# set(OpenCV_DIR "/home/neosentinel/Projects/TensorRT-For-YOLO-Series/cpp/opencv-4.x/build")
# include_directories(${PROJECT_SOURCE_DIR}/include)
# # include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# # cuda
# include_directories(/usr/local/cuda/include)
# link_directories(/usr/local/cuda/lib64)
# # tensorrt
# include_directories(/usr/include/x86_64-linux-gnu/)
# link_directories(/usr/lib/x86_64-linux-gnu/)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})

# add_executable(yolo ${PROJECT_SOURCE_DIR}/main.cpp)
# target_link_libraries(yolo nvinfer)
# target_link_libraries(yolo nvinfer_plugin)
# target_link_libraries(yolo cudart)

# target_link_libraries(yolo ${OpenCV_LIBS})

# add_definitions(-O2 -pthread)



cmake_minimum_required(VERSION 2.6)

project(yolo)

# C++ Standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Set OpenCV Path (only needed if OpenCV is custom installed)
set(OpenCV_DIR "/home/neosentinel/Projects/TensorRT-For-YOLO-Series/cpp/opencv-4.x/build")
find_package(OpenCV REQUIRED)

# CUDA settings
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)

# TensorRT settings (if installed in a custom path)
include_directories(/TensorRT-10.0.1.6/include/)  # Adjust if needed
link_directories(/TensorRT-10.0.1.6/include/)          # Adjust if needed

# CUDA and TensorRT includes and libraries
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARIES})
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add additional definitions and flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# OpenCV include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable and link libraries
add_executable(yolo ${PROJECT_SOURCE_DIR}/main.cpp)
target_link_libraries(yolo nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

# Optionally, add these for extra flags (if needed)
add_definitions(-O2 -pthread)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GSTREAMER REQUIRED gstreamer-1.0)

include_directories(${GSTREAMER_INCLUDE_DIRS})
link_directories(${GSTREAMER_LIBRARY_DIRS})

target_link_libraries(yolo ${GSTREAMER_LIBRARIES})

