list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkRenderer.h
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}"
    PREFIX "EnigneSystems"
    FILES ${CURRENT_DIR_SOURCES})

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})
target_include_directories(Spark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})