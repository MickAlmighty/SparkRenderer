#ifndef ENUMS_H
#define ENUMS_H

namespace spark {

enum class ShaderType : uint16_t
{
	DEFAULT_SHADER = 0,
	POSTPROCESSING_SHADER,
	SCREEN_SHADER
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