#ifndef ENUMS_H
#define ENUMS_H

namespace spark
{
enum class ShaderType : unsigned char
{
    DEFAULT_SHADER = 0,
    LIGHT_SHADER = 1,
    POSTPROCESSING_SHADER = 2,
    SCREEN_SHADER = 3,
    MOTION_BLUR_SHADER = 4,
    EQUIRECTANGULAR_TO_CUBEMAP_SHADER = 5,
    CUBEMAP_SHADER = 6,
    IRRADIANCE_SHADER = 7,
    PREFILTER_SHADER = 8,
    BRDF_SHADER = 9,
    BRIGHT_PASS_SHADER = 10,
    DOWNSCALE_SHADER = 11,
    GAUSSIAN_BLUR_SHADER = 12,
    LIGHT_SHAFTS_SHADER = 13,
    SSAO_SHADER = 14
};

enum class TextureTarget : unsigned char
{
    DIFFUSE_TARGET = 1,
    NORMAL_TARGET = 2,
    ROUGHNESS_TARGET = 3,
    METALNESS_TARGET = 4
};

}  // namespace spark
#endif