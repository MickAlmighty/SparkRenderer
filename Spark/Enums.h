#ifndef ENUMS_H
#define ENUMS_H

namespace spark {

enum class ShaderType : uint16_t
{
	DEFAULT_SHADER = 0,
	LIGHT_SHADER,
	POSTPROCESSING_SHADER,
	SCREEN_SHADER,
	MOTION_BLUR_SHADER,
	EQUIRECTANGULAR_TO_CUBEMAP_SHADER,
	CUBEMAP_SHADER,
	IRRADIANCE_SHADER,
	PREFILTER_SHADER,
	BRDF_SHADER,
	PATH_SHADER,
	BRIGHT_PASS_SHADER,
	DOWNSCALE_SHADER,
	GAUSSIAN_BLUR_SHADER
};

enum class TextureTarget : uint16_t
{
	DIFFUSE_TARGET = 1,
	NORMAL_TARGET,
	ROUGHNESS_TARGET,
	METALNESS_TARGET
};

}
#endif