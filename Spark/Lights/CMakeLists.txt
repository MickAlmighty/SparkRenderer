list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/DirectionalLight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DirectionalLight.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LightManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbe.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LightStatus.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PointLight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PointLight.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SpotLight.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SpotLight.h
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})
target_include_directories(Spark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})