list(APPEND SparkSources
    ${CMAKE_CURRENT_SOURCE_DIR}/Buffer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Camera.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Clock.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Cube.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Clock.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedLightCullingPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ClusterBasedLightCullingPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CommonUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/CommonUtils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Component.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Component.h
    ${CMAKE_CURRENT_SOURCE_DIR}/DeferredRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/DeferredRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Enums.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Enums.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GameObject.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GameObject.h
    ${CMAKE_CURRENT_SOURCE_DIR}/GBuffer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GBuffer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/glad_glfw3.h
    ${CMAKE_CURRENT_SOURCE_DIR}/glad.c
    ${CMAKE_CURRENT_SOURCE_DIR}/IObserver.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/JsonSerializer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/JsonSerializer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbesRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LightProbesRenderer.hpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/RenderingRequest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Resource.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Resource.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceFactory.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceFactory.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceIdentifier.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceIdentifier.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLibrary.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLibrary.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLoader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ResourceLoader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Scene.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ScreenQuad.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Shader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Shader.h
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderDescriptor.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderDescriptor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderParser.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShaderParser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCreator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ShapeCreator.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Skybox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Skybox.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Spark.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Spark.h
    ${CMAKE_CURRENT_SOURCE_DIR}/SparkConfig.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stb_image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Structs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Structs.h
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedDeferredRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedDeferredRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedForwardPlusRenderer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedForwardPlusRenderer.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedLightCullingPass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TileBasedLightCullingPass.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Texture.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.h
    ${CMAKE_CURRENT_SOURCE_DIR}/Timer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WorldTransform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/WorldTransform.h
)

add_library(Spark "")
target_sources(Spark PRIVATE ${SparkSources})

add_subdirectory(EngineSystems)
add_subdirectory(GUI)
add_subdirectory(HID)
add_subdirectory(lights)
add_subdirectory(effects)

# Create the source groups for source tree with root at CMAKE_CURRENT_SOURCE_DIR.
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SparkSources})

target_include_directories(Spark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/include)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_definitions(DEBUG)
endif ()

# message("Paths to libraries ${LIB_PATH}")
target_link_directories(Spark PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/lib/ReleaseLib ${CMAKE_SOURCE_DIR}/thirdparty/lib/DebugLib)
target_link_directories(Spark PUBLIC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

#GLFW
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
FetchContent_Declare(
        glfw
        GIT_REPOSITORY "https://github.com/glfw/glfw"
        GIT_TAG "814b7929c5add4b0541ccad26fb81f28b71dc4d8"
)
FetchContent_MakeAvailable(glfw)

#SPDLOG
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY "https://github.com/gabime/spdlog"
        GIT_TAG "100f30043f33277122e0991c83845a2617172ffd"
)
FetchContent_MakeAvailable(spdlog)

# #RTTR
# FetchContent_Declare(
#         rttr
#         GIT_REPOSITORY "https://github.com/Doctor-Ned/rttr"
#         GIT_TAG "a851ebae2fc8f08f05a84d358786d7f7c5856c82"
# )

# FetchContent_MakeAvailable(rttr)
# message("rttr source dir: ${rttr_SOURCE_DIR}/scr")
# message("rttr source dir: ${rttr_BINARY_DIR}/scr")
# target_include_directories(Spark PUBLIC ${rttr_SOURCE_DIR}/src)
# target_include_directories(Spark PUBLIC ${rttr_BINARY_DIR}/src)

#JSONCPP
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE INTERNAL "Turn off tests")
set(JSONCPP_WITH_TESTS OFF CACHE INTERNAL "Turn off tests")

FetchContent_Declare(
        jsoncpp
        GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp"
        GIT_TAG "1.9.4"
)
FetchContent_MakeAvailable(jsoncpp)
message("jsoncpp source dir: ${jsoncpp_SOURCE_DIR}/include")
target_include_directories(Spark PUBLIC ${jsoncpp_SOURCE_DIR}/include)

#ASSIMP
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "Turn off tests")
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")

FetchContent_Declare(
        assimp
        GIT_REPOSITORY "https://github.com/assimp/assimp"
        GIT_TAG "v5.0.1"
)
FetchContent_MakeAvailable(assimp)

if(MSVC)
    target_link_libraries(Spark debug librttr_core_d optimized librttr_core)
else()
    target_link_libraries(Spark X11.so)
    target_link_libraries(Spark librttr_core.a)
    target_link_libraries(Spark ${CMAKE_DL_LIBS})
endif()
target_link_libraries(Spark assimp)
target_link_libraries(Spark debug jsoncppd optimized jsoncpp)
target_link_libraries(Spark glfw)
target_link_libraries(Spark spdlog)
