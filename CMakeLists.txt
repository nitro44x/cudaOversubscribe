cmake_minimum_required(VERSION 3.20.0)

project(oversub LANGUAGES CXX CUDA)

option(BUILD_DOCS "Build documentation" OFF)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CUDA_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(tpl_src
    thirdparty/CLI11.hpp
)

set(srcs 
    src/main.cpp
    src/oversubTest.hpp
    src/Array.hpp
    src/kernels.hpp
)

set(cuda_srcs 
    src/oversubTest.cpp 
    src/Array.cpp
    src/kernels.cpp
)

set(docs README.md)

set_property(SOURCE ${cuda_srcs} PROPERTY LANGUAGE CUDA)

add_executable(oversub ${srcs} ${cuda_srcs} ${docs} ${tpl_src})
target_include_directories(oversub PRIVATE thirdparty)

if(BUILD_DOCS)
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
    set(DOXYGEN_HTML_EXTRA_STYLESHEET docs/doxygen-awesome.css 
                                      docs/doxygen-awesome-sidebar-only.css 
                                      docs/doxygen-awesome-sidebar-only-darkmode-toggle.css
                                      docs/custom.css
                                      )
    set(DOXYGEN_HTML_FOOTER docs/footer.html)
    set(DOXYGEN_HTML_HEADER docs/header.html)
    set(DOXYGEN_GENERATE_TREEVIEW YES)
    set(DOXYGEN_HTML_EXTRA_FILES docs/doxygen-awesome-darkmode-toggle.js)

    doxygen_add_docs(
        doxygen
        src ${docs}
        COMMENT "Generate html docs"
    )
endif()

find_program(CLANGFORMAT clang-format)
if(CLANGFORMAT)
    file(GLOB_RECURSE ALL_SOURCE_FILES src/*.h src/*.hpp src/*.cpp src/*.c)

    add_custom_target(
            fmt
            COMMAND ${CLANGFORMAT}
            -style=file
            -i
            ${ALL_SOURCE_FILES}
    )
endif()
