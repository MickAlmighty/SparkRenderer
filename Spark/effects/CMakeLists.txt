list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/AmbientOcclusion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AmbientOcclusion.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BloomPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BloomPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlurPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/BlurPass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DepthOfFieldPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DepthOfFieldPass.h
    ${CMAKE_CURRENT_SOURCE_DIR}/FxaaPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/FxaaPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightShaftsPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightShaftsPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MotionBlurPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MotionBlurPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessingStack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PostProcessingStack.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SkyboxPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SkyboxPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TexturePass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TexturePass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ToneMapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ToneMapper.hpp
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})