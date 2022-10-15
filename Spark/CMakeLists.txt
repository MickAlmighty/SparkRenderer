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
    ${CMAKE_CURRENT_SOURCE_DIR}/glad.c
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

target_include_directories(Spark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(Spark PUBLIC ${CMAKE_SOURCE_DIR}/thirdparty/include)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    #target_compile_definitions(Spark PRIVATE DEBUG)
    if(MSVC_VERSION GREATER_EQUAL "1930")
        target_compile_options(Spark PRIVATE "/ZI")
    endif()
endif()
target_compile_definitions(Spark PRIVATE GLM_FORCE_ALIGNED_GENTYPES)

target_link_directories(Spark PUBLIC ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

#GLFW
FetchContent_Declare(
        glfw
        GIT_REPOSITORY "https://github.com/glfw/glfw"
        GIT_TAG "814b7929c5add4b0541ccad26fb81f28b71dc4d8"
)
set(GLFW_BUILD_EXAMPLES OFF CACHE INTERNAL "Build the GLFW example programs")
set(GLFW_BUILD_TESTS OFF CACHE INTERNAL "Build the GLFW test programs")
set(GLFW_BUILD_DOCS OFF CACHE INTERNAL "Build the GLFW documentation")
FetchContent_MakeAvailable(glfw)

#SPDLOG
FetchContent_Declare(
        spdlog
        GIT_REPOSITORY "https://github.com/gabime/spdlog"
        GIT_TAG "100f30043f33277122e0991c83845a2617172ffd"
)
FetchContent_MakeAvailable(spdlog)

#RTTR
FetchContent_Declare(
        rttr
        GIT_REPOSITORY "https://github.com/MickAlmighty/rttr"
        GIT_TAG "4e9f49a5e67f569331d706744064e7bae5796306" #15/05/2021
)
set(BUILD_RTTR_DYNAMIC OFF CACHE INTERNAL "" FORCE)
set(BUILD_STATIC ON CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(rttr)
target_include_directories(Spark PUBLIC ${rttr_SOURCE_DIR}/src)
target_include_directories(Spark PUBLIC ${rttr_BINARY_DIR}/src)
target_link_directories(Spark PUBLIC ${rttr_BINARY_DIR}/lib)

#STB
FetchContent_Declare(
        stb
        GIT_REPOSITORY "https://github.com/nothings/stb"
        GIT_TAG "8b5f1f37b5b75829fc72d38e7b5d4bcbf8a26d55"
)
FetchContent_GetProperties(stb)
if(NOT stb_POPULATED)
  FetchContent_Populate(stb)
  target_include_directories(Spark PUBLIC ${stb_SOURCE_DIR})
endif()

#TINYGLTF
FetchContent_Declare(
        tinygltf
        GIT_REPOSITORY "https://github.com/syoyo/tinygltf"
        GIT_TAG "19a41d20ec03d35cee68ff85511bc5875f6037b8"
)
set(TINYGLTF_HEADER_ONLY ON CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(tinygltf)
target_include_directories(Spark PUBLIC ${tinygltf_SOURCE_DIR})

#JSONCPP
FetchContent_Declare(
        jsoncpp
        GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp"
        GIT_TAG "1.9.5"
)
set(JSONCPP_WITH_TESTS OFF CACHE INTERNAL "Turn off tests" FORCE)
set(JSONCPP_WITH_POST_BUILD_UNITTEST OFF CACHE INTERNAL "Turn off tests" FORCE)
set(JSONCPP_WITH_PKGCONFIG_SUPPORT OFF CACHE INTERNAL "Turn off tests" FORCE)
set(JSONCPP_WITH_CMAKE_PACKAGE OFF CACHE INTERNAL "Turn off tests" FORCE)
set(JSONCPP_WITH_TESTS OFF CACHE INTERNAL "Turn off tests" FORCE)
set(JSONCPP_WITH_STRICT_ISO OFF CACHE INTERNAL "Turn off tests" FORCE)
FetchContent_MakeAvailable(jsoncpp)
target_include_directories(Spark PUBLIC ${jsoncpp_SOURCE_DIR}/include)

#ASSIMP
FetchContent_Declare(
        assimp
        GIT_REPOSITORY "https://github.com/assimp/assimp"
        GIT_TAG "v5.2.5"
)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "If the supplementary tools for Assimp are built in addition to the library.")
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "Turn off tests")
set(ASSIMP_INSTALL OFF CACHE INTERNAL "Disable this if you want to use assimp as a submodule.")
set(ASSIMP_NO_EXPORT ON CACHE INTERNAL "Disable Assimp's export functionality.")
FetchContent_MakeAvailable(assimp)

#spirv-headers (NEEDED ONLY FOR SPIRV-TOOLS)
FetchContent_Declare(
        spirv-headers
        GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Headers"
        GIT_TAG "sdk-1.3.224.1"
)
set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE INTERNAL "" FORCE)
set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(spirv-headers)

#spirv-tools (NEEDED ONLY FOR SHADERC)
FetchContent_Declare(
        spirv-tools
        GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Tools"
        GIT_TAG "v2022.3"
)
set(SPIRV_SKIP_EXECUTABLES ON CACHE INTERNAL "" FORCE)
set(SPIRV_SKIP_TESTS ON CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(spirv-tools)

#spirv-headers (NEEDED ONLY FOR SHADERC)
FetchContent_Declare(
        glslang
        GIT_REPOSITORY "https://github.com/KhronosGroup/glslang"
        GIT_TAG "11.12.0"
)
set(SKIP_GLSLANG_INSTALL ON CACHE INTERNAL "" FORCE)
set(ENABLE_GLSLANG_BINARIES OFF CACHE INTERNAL "" FORCE)
set(ENABLE_GLSLANG_INSTALL OFF CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(glslang)

#SHADERC
FetchContent_Declare(
        shaderc
        GIT_REPOSITORY "https://github.com/google/shaderc"
        GIT_TAG "v2022.3"
)
set(SHADERC_SKIP_TESTS ON CACHE INTERNAL "" FORCE)
set(SHADERC_SKIP_INSTALL ON CACHE BOOL "" FORCE)
set(SHADERC_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
set(SHADERC_SKIP_COPYRIGHT_CHECK ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(shaderc)

#SPIRV_CROSS
FetchContent_Declare(
        spirv_cross
        GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Cross"
        GIT_TAG "sdk-1.3.224.1"
)
set(SPIRV_CROSS_CLI OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_SKIP_INSTALL ON CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_MSL OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_CPP OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE INTERNAL "" FORCE)
set(SPIRV_CROSS_ENABLE_C_API OFF CACHE INTERNAL "" FORCE)
FetchContent_MakeAvailable(spirv_cross)
target_include_directories(Spark PUBLIC ${spirv_cross_SOURCE_DIR})

#GLM
FetchContent_Declare(
        glm
        GIT_REPOSITORY "https://github.com/g-truc/glm"
        GIT_TAG "0.9.9.8"
)
FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
  target_include_directories(Spark PUBLIC ${glm_SOURCE_DIR})
endif()

#GLI
FetchContent_Declare(
        gli
        GIT_REPOSITORY "https://github.com/g-truc/gli"
        GIT_TAG "779b99ac6656e4d30c3b24e96e0136a59649a869" #15/05/2021
)
FetchContent_GetProperties(gli)
if(NOT gli_POPULATED)
  FetchContent_Populate(gli)
  target_include_directories(Spark PUBLIC ${gli_SOURCE_DIR})
endif()

if(MSVC)
    target_link_libraries(Spark debug librttr_core_d optimized librttr_core_s)
else()
    target_link_libraries(Spark X11.so)
    target_link_libraries(Spark debug rttr_core_d optimized rttr_core)
endif()
target_link_libraries(Spark assimp)
target_link_libraries(Spark jsoncpp)
target_link_libraries(Spark glfw)
target_link_libraries(Spark spdlog)
target_link_libraries(Spark shaderc)
target_link_libraries(Spark spirv-cross-core spirv-cross-reflect spirv-cross-glsl)
