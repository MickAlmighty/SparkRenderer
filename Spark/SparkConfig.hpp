#pragma once
#include <rttr/registration>

namespace spark
{
struct SparkConfig final
{
    SparkConfig();

    unsigned int width{1280};
    unsigned int height{720};
    std::string pathToResources{R"(res)"};
    bool vsync{true};
};
}  // namespace spark