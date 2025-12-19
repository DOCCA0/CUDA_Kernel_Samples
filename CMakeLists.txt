cmake_minimum_required(VERSION 3.18)
project(cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 11)

# set(CMAKE_CUDA_ARCHITECTURES 89)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

add_executable(add ./elementwise/add.cu)

# set_target_properties(add PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON
# )

# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()
