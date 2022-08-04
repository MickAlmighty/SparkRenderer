list(APPEND CURRENT_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GltfLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GltfLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ILoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderLoader.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TextureLoader.hpp
)

target_sources(Spark PRIVATE ${CURRENT_DIR_SOURCES})