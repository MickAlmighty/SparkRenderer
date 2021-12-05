#include "SparkConfig.hpp"

namespace spark
{
SparkConfig::SparkConfig()
{
}
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