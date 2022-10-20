add_executable(SparkExec main.cpp)

target_include_directories(SparkExec PUBLIC ${CMAKE_SOURCE_DIR}/Spark)

if (MSVC_VERSION GREATER_EQUAL "1900")
    #https://github.com/headupinclouds/cmake_whole_archive
    target_link_libraries(SparkExec Spark -WHOLEARCHIVE:$<TARGET_FILE:Spark>)
else()
    target_link_libraries(SparkExec -Wl,--whole-archive Spark -Wl,--no-whole-archive)
endif()