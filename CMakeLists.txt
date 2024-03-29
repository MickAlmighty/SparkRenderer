﻿cmake_minimum_required (VERSION 3.19)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0077 NEW)

# ---------------------------------------------------------------------------
# use ccache if found, has to be done before project()
# ---------------------------------------------------------------------------
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    message(STATUS "use ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_FOUND}")
endif()

project (SparkRenderer VERSION 1.0)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if (MSVC_VERSION GREATER_EQUAL "1900")
    message("MSVC_VERSION ${MSVC_VERSION}")
    include(CheckCXXCompilerFlag)
    # CHECK_CXX_COMPILER_FLAG("/std:c++latest" _cpp_latest_flag_supported)
    # if (_cpp_latest_flag_supported)
    #     add_compile_options("/std:c++latest")
    # endif()
    CHECK_CXX_COMPILER_FLAG("/std:c++17" _cpp17_flag_supported)
    if (_cpp17_flag_supported)
        add_compile_options("/std:c++17")
    endif()
    CHECK_CXX_COMPILER_FLAG("/MP" _multithreading)
    if (_multithreading)
        add_compile_options("/MP")
    endif()
    add_compile_options("/bigobj")
else()
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(dependencies)
add_subdirectory(Spark)
add_subdirectory(SparkExec)
add_subdirectory(SparkTests)