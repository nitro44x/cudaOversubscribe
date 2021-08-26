cmake_minimum_required(VERSION 3.20.0)

project(oversub LANGUAGES CXX CUDA)

set(srcs 
    src/main.cpp
    src/CLI11.hpp
    src/kernel.hpp 
)

set(cuda_srcs 
    src/kernel.cpp 
)

set(docs README.md)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_executable(oversub ${srcs} ${cuda_srcs} ${docs})