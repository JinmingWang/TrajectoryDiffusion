cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(preprocess)

set(Torch_DIR /home/jimmy/anaconda3/envs/yolov5/lib/python3.11/site-packages/torch/share/cmake/Torch/)

include(cmake/cuda.cmake)
message("cuda include dir" ${CUDA_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# TO import package, the following file is changed:
# /home/jimmy/anaconda3/envs/yolov5/lib/python3.11/site-packages/torch/share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA.cmake
# CUDA path is set manually, if you want to find where is changed, search for # jimmy
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
message(STATUS "CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "Pytorch status:")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")


find_package(OpenMP REQUIRED)
message(STATUS "OpenMP status:")
message(STATUS "    OpenMP_CXX_FLAGS: ${OpenMP_CXX_FLAGS}")

add_executable(preprocess preprocess.cpp)
target_link_libraries(preprocess "${TORCH_LIBRARIES}" OpenMP::OpenMP_CXX)
set_property(TARGET preprocess PROPERTY CXX_STANDARD 17)