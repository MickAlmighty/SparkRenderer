list(APPEND SparkSources
    ${CMAKE_CURRENT_SOURCE_DIR}/Animation.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Animation.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationCreator.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationData.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationData.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationPlayer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/AnimationPlayer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CameraManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CameraManager.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EditorCamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/EditorCamera.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Clock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Cube.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Clock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Component.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Enums.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Enums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GameObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GameObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/glad_glfw3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/IObserver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ICamera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ICamera.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/JsonSerializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/JsonSerializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LocalTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LocalTransform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Logging.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Logging.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Mesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/MeshPlane.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/MeshPlane.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelMesh.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ModelMesh.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Observable.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenGLContext.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/OpenGLContext.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ReflectionUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/PbrCubemapTexture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/PbrCubemapTexture.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Resource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Resource.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceIdentifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceIdentifier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLibrary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Scene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneManager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SceneManager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ScreenQuad.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Shader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderDescriptor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderDescriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCreator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Skybox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Skybox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Spark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Spark.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkConfig.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkConfig.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stb_image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/VertexAttribute.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Transform.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WorldTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WorldTransform.h
)

add_library(Spark "")
target_sources(Spark PRIVATE ${SparkSources})

add_subdirectory(GUI)
add_subdirectory(HID)
add_subdirectory(lights)
add_subdirectory(effects)
add_subdirectory(renderers)
add_subdirectory(scripts)
add_subdirectory(utils)
add_subdirectory(loaders)

# Create the source groups for source tree with root at CMAKE_CURRENT_SOURCE_DIR.
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SparkSources})

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #target_compile_definitions(Spark PRIVATE DEBUG)
    if(MSVC_VERSION GREATER_EQUAL "1930")
        target_compile_options(Spark PRIVATE "/ZI")
    endif()
endif()
target_compile_definitions(Spark PRIVATE GLM_FORCE_ALIGNED_GENTYPES)

target_include_directories(Spark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${GLAD_INCLUDE_DIRS})
target_include_directories(Spark PUBLIC ${JSONCPP_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${GLM_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${GLI_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${TINYGLTF_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${STB_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${SPIRV_CROSS_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${RTTR_SOURCE_DIR}/src)
target_include_directories(Spark PUBLIC ${RTTR_BINARY_DIR}/src)
target_include_directories(Spark PUBLIC ${DIRENT_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${IMGUI_SOURCE_DIR})


target_link_directories(Spark PUBLIC ${RTTR_BINARY_DIR}/lib)
target_link_directories(Spark PUBLIC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
if(MSVC)
    target_link_libraries(Spark debug librttr_core_d optimized librttr_core)
else()
    target_link_libraries(Spark X11.so)
    target_link_libraries(Spark debug rttr_core_d optimized rttr_core)
endif()
target_link_libraries(Spark assimp)
target_link_libraries(Spark jsoncpp)
target_link_libraries(Spark glad)
target_link_libraries(Spark glfw)
target_link_libraries(Spark spdlog)
target_link_libraries(Spark shaderc)
target_link_libraries(Spark spirv-cross-core spirv-cross-reflect spirv-cross-glsl)