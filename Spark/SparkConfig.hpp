#pragma once
#include <rttr/registration>

namespace spark
{
struct SparkConfig final
{
    SparkConfig();

    unsigned int width{1920};
    unsigned int height{1080};
    std::string pathToResources{R"(res)"};
    bool vsync{false};
};
}  // namespace spark