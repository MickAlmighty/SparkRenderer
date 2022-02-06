add_executable(SparkExec main.cpp)

target_include_directories(SparkExec PUBLIC ${CMAKE_SOURCE_DIR}/Spark)

if (MSVC_VERSION GREATER_EQUAL "1900")
    #https://github.com/headupinclouds/cmake_whole_archive
    target_link_libraries(SparkExec Spark -WHOLEARCHIVE:$<TARGET_FILE:Spark>)
else()
    target_link_libraries(SparkExec -Wl,--whole-archive Spark -Wl,--no-whole-archive)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    file(GLOB DYNAMIC_LIBS  ${CMAKE_SOURCE_DIR}/thirdparty/lib/ReleaseLib/*.dll
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/Commonlib/*.dll
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/ReleaseLib/*.so
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/Commonlib/*.so)
else()
    file(GLOB DYNAMIC_LIBS  ${CMAKE_SOURCE_DIR}/thirdparty/lib/DebugLib/*.dll
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/Commonlib/*.dll
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/DebugLib/*.so
                            ${CMAKE_SOURCE_DIR}/thirdparty/lib/Commonlib/*.so)
endif()

message("Dynamic Libraries: ${DYNAMIC_LIBS}") 

foreach(DynamicLib ${DYNAMIC_LIBS})
    add_custom_command(TARGET SparkExec PRE_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        ${DynamicLib} $<TARGET_FILE_DIR:SparkExec>)
endforeach()