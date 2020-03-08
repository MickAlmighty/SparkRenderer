#include "Enums.h"
#include <rttr/registration>

RTTR_REGISTRATION
{
    rttr::registration::enumeration<spark::ShaderType>("ShaderType")(
        rttr::value("DEFAULT_SHADER", spark::ShaderType::DEFAULT_SHADER), rttr::value("LIGHT_SHADER", spark::ShaderType::LIGHT_SHADER),
        rttr::value("POSTPROCESSING_SHADER", spark::ShaderType::TONE_MAPPING_SHADER),
        rttr::value("SCREEN_SHADER", spark::ShaderType::SCREEN_SHADER), rttr::value("MOTION_BLUR_SHADER", spark::ShaderType::MOTION_BLUR_SHADER),
        rttr::value("EQUIRECTANGULAR_TO_CUBEMAP_SHADER", spark::ShaderType::EQUIRECTANGULAR_TO_CUBEMAP_SHADER),
        rttr::value("CUBEMAP_SHADER", spark::ShaderType::CUBEMAP_SHADER), rttr::value("IRRADIANCE_SHADER", spark::ShaderType::IRRADIANCE_SHADER),
        rttr::value("PREFILTER_SHADER", spark::ShaderType::PREFILTER_SHADER), rttr::value("BRDF_SHADER", spark::ShaderType::BRDF_SHADER));

    rttr::registration::enumeration<spark::TextureTarget>("TextureTarget")(rttr::value("DIFFUSE_TARGET", spark::TextureTarget::DIFFUSE_TARGET),
                                                                           rttr::value("NORMAL_TARGET", spark::TextureTarget::NORMAL_TARGET),
                                                                           rttr::value("ROUGHNESS_TARGET", spark::TextureTarget::ROUGHNESS_TARGET),
                                                                           rttr::value("METALNESS_TARGET", spark::TextureTarget::METALNESS_TARGET));
}