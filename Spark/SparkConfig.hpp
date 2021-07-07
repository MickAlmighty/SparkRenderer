#pragma once
#include <rttr/registration>

namespace spark
{
struct SparkConfig final
{
    unsigned int width{1280};
    unsigned int height{720};
    std::string pathToResources{R"(res)"};
    bool vsync{true};
};
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::SparkConfig>("SparkConfig")
        .constructor()(rttr::policy::ctor::as_object)
        .property("width", &spark::SparkConfig::width)
        .property("height", &spark::SparkConfig::height)
        .property("pathToResources", &spark::SparkConfig::pathToResources)
        .property("vsync", &spark::SparkConfig::vsync);
}