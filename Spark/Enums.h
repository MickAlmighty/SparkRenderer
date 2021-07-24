#pragma once

namespace spark
{
enum class ShaderType : unsigned char
{
    PBR = 0,
    COLOR_ONLY
};

enum class TextureTarget : unsigned char
{
    DIFFUSE_TARGET = 1,
    NORMAL_TARGET = 2,
    ROUGHNESS_TARGET = 3,
    METALNESS_TARGET = 4,
    HEIGHT_TARGET = 5,
    AO_TARGET = 6
};

}  // namespace spark