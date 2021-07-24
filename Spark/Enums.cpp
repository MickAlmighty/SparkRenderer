#include "Enums.h"
#include <rttr/registration>

RTTR_REGISTRATION
{
    rttr::registration::enumeration<spark::ShaderType>("ShaderType")(
        rttr::value("DEFAULT_SHADER", spark::ShaderType::PBR), rttr::value("LIGHT_SHADER", spark::ShaderType::COLOR_ONLY));

    rttr::registration::enumeration<spark::TextureTarget>("TextureTarget")(rttr::value("DIFFUSE_TARGET", spark::TextureTarget::DIFFUSE_TARGET),
                                                                           rttr::value("NORMAL_TARGET", spark::TextureTarget::NORMAL_TARGET),
                                                                           rttr::value("ROUGHNESS_TARGET", spark::TextureTarget::ROUGHNESS_TARGET),
                                                                           rttr::value("METALNESS_TARGET", spark::TextureTarget::METALNESS_TARGET),
                                                                           rttr::value("HEIGHT_TARGET", spark::TextureTarget::HEIGHT_TARGET),
                                                                           rttr::value("AO_TARGET", spark::TextureTarget::AO_TARGET));
}