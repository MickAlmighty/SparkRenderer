list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedDeferredRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedDeferredRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedLightCullingPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedLightCullingPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeferredRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeferredRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GBuffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbesRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbesRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedDeferredRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedDeferredRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedLightCullingPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedLightCullingPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderingRequest.h
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})