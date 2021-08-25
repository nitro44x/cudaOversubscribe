cmake_minimum_required(VERSION 3.20.0)

project(oversub LANGUAGES CXX CUDA)

add_executable(oversub main.cpp kernel.hpp kernel.cu)