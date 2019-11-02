#ifndef ENUMS_H
#define ENUMS_H

namespace spark {

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
	BRDF_SHADER = 9
};

enum class TextureTarget : unsigned char
{
	DIFFUSE_TARGET = 0,
	NORMAL_TARGET = 1,
	ROUGHNESS_TARGET = 2,
	METALNESS_TARGET = 3
};

}
#endif