cmake_minimum_required(VERSION 3.20.0)

project(oversub LANGUAGES CXX CUDA)

option(BUILD_DOCS "Build documentation" ON)

set(tpl_src
    thirdparty/CLI11.hpp
)

set(srcs 
    src/main.cpp
    src/oversubTest.hpp
    src/Array.hpp
)

set(cuda_srcs 
    src/oversubTest.cpp 
)

set(docs README.md)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_executable(oversub ${srcs} ${cuda_srcs} ${docs} ${tpl_src})
target_include_directories(oversub PRIVATE thirdparty)

if(BUILD_DOCS)
    find_package(Doxygen)

    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)

    doxygen_add_docs(
        doxygen
        src ${docs}
        COMMENT "Generate html docs"
    )
endif()