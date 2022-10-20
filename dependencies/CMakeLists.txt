include(FetchContent)

#GLFW
FetchContent_Declare(
    glfw
    GIT_REPOSITORY "https://github.com/glfw/glfw"
    GIT_TAG "814b7929c5add4b0541ccad26fb81f28b71dc4d8"
)

#SPDLOG
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY "https://github.com/gabime/spdlog"
    GIT_TAG "100f30043f33277122e0991c83845a2617172ffd"
)

#RTTR
FetchContent_Declare(
    rttr
    GIT_REPOSITORY "https://github.com/MickAlmighty/rttr"
    GIT_TAG "4e9f49a5e67f569331d706744064e7bae5796306" #15/05/2021
)

#STB
FetchContent_Declare(
    stb
    GIT_REPOSITORY "https://github.com/nothings/stb"
    GIT_TAG "8b5f1f37b5b75829fc72d38e7b5d4bcbf8a26d55"
)

#TINYGLTF
FetchContent_Declare(
    tinygltf
    GIT_REPOSITORY "https://github.com/syoyo/tinygltf"
    GIT_TAG "19a41d20ec03d35cee68ff85511bc5875f6037b8"
)

#JSONCPP
FetchContent_Declare(
    jsoncpp
    GIT_REPOSITORY "https://github.com/open-source-parsers/jsoncpp"
    GIT_TAG "1.9.5"
)

#ASSIMP
FetchContent_Declare(
    assimp
    GIT_REPOSITORY "https://github.com/assimp/assimp"
    GIT_TAG "v5.2.5"
)

#spirv-headers (NEEDED ONLY FOR SPIRV-TOOLS)
FetchContent_Declare(
    spirv-headers
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Headers"
    GIT_TAG "sdk-1.3.224.1"
)

#spirv-tools (NEEDED ONLY FOR SHADERC)
FetchContent_Declare(
    spirv-tools
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Tools"
    GIT_TAG "v2022.3"
)

#spirv-headers (NEEDED ONLY FOR SHADERC)
FetchContent_Declare(
    glslang
    GIT_REPOSITORY "https://github.com/KhronosGroup/glslang"
    GIT_TAG "11.12.0"
)

#SHADERC
FetchContent_Declare(
    shaderc
    GIT_REPOSITORY "https://github.com/google/shaderc"
    GIT_TAG "v2022.3"
)

#SPIRV_CROSS
FetchContent_Declare(
    spirv_cross
    GIT_REPOSITORY "https://github.com/KhronosGroup/SPIRV-Cross"
    GIT_TAG "sdk-1.3.224.1"
)

#GLM
FetchContent_Declare(
    glm
    GIT_REPOSITORY "https://github.com/g-truc/glm"
    GIT_TAG "0.9.9.8"
)

#GLI
FetchContent_Declare(
    gli
    GIT_REPOSITORY "https://github.com/g-truc/gli"
    GIT_TAG "779b99ac6656e4d30c3b24e96e0136a59649a869" #15/05/2021
)

#DIRENT
FetchContent_Declare(
    dirent
    GIT_REPOSITORY "https://github.com/tronkko/dirent"
    GIT_TAG "1.23.2"
)

#IMGUI
FetchContent_Declare(
    imgui
    GIT_REPOSITORY "https://github.com/ocornut/imgui"
    GIT_TAG "v1.88"
)

FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/Dav1dde/glad.git
  GIT_TAG        v0.1.36
)

add_subdirectory(glad)
add_subdirectory(glfw)
add_subdirectory(spdlog)
add_subdirectory(rttr)
add_subdirectory(stb)
add_subdirectory(tinygltf)
add_subdirectory(jsoncpp)
add_subdirectory(assimp)
add_subdirectory(spirv-headers)
add_subdirectory(spirv-tools)
add_subdirectory(glslang)
add_subdirectory(shaderc)
add_subdirectory(spirv_cross)
add_subdirectory(glm)
add_subdirectory(gli)
add_subdirectory(dirent)
add_subdirectory(imgui)