list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/CommonUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CommonUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FileUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FileUtils.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GlHandle.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderCompiler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderCompiler.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderParser.hpp
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})