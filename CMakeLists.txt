cmake_minimum_required(VERSION 2.6.0 FATAL_ERROR)
project(nlfilter CXX C)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/etc)

find_package(CUDA REQUIRED)
find_package(FLTK REQUIRED)
find_package(OpenGL REQUIRED)
#find_package(Glew)

find_program(GCC NAMES gcc-4.5 gcc-4.5.3 gcc-4.5.2 gcc-4.5.1 gcc-4.5.0 gcc-4.4
    gcc-4.4.7 gcc-4.4.6 gcc-4.4.5 gcc-4.4.4 gcc-4.4.3 gcc-4.4.2 gcc-4.4.1
    gcc-4.4.0 gcc-4.3 gcc-4.3.6 gcc-4.3.5 gcc-4.3.4 gcc-4.3.3 gcc-4.3.2
    gcc-4.3.1 gcc-4.3.0 gcc)

if(NOT GCC)
    message(FATAL_ERROR "Cannot find suitable compiler, must be gcc < 4.6")
endif()

message(STATUS "Using ${GCC}")

#if(NOT CUDA_SDK_ROOT_DIR)
#    message(FATAL_ERROR "Please set environment variable NVSDKCUDA_ROOT to the
#    SDK base directory (it contains 'common','lib','bin', etc), for instance
#    '/usr/local/cuda/NVIDIA_GPU_Computing_SDK/C'")
#endif()

#find_library(CUTIL_LIBRARY NAMES cutil cutil_x86_64 PATHS ${CUDA_SDK_ROOT_DIR}/lib)

set(CUDA_SM 20 CACHE bool "Compile for which sm?")

set(CUDA_NVCC_FLAGS
  # -ccbin=${GCC}
  #-Xptxas -dlcm=cg # gmem uses only L2 cache
  #-Xptxas -dlcm=ca # gmem uses both L1 and L2 cache (default)
  --ptxas-options=-v 
  -ftz=true -prec-div=false -prec-sqrt=false
  #--opencc-options -OPT:Olimit=0
)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} 
    -gencode arch=compute_${CUDA_SM},code=sm_${CUDA_SM})

add_subdirectory(src)
