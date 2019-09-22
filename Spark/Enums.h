#ifndef ENUMS_H
#define ENUMS_H

enum class ShaderType
{
	DEFAULT_SHADER = 0,
	POSTPROCESSING_SHADER,
	SCREEN_SHADER
};

enum class TextureTarget
{
	DIFFUSE_TARGET = 1,
	NORMAL_TARGET
};
#endif