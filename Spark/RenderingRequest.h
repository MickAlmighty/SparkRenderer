#pragma once

#include <memory>

#include "Enums.h"
#include "GameObject.h"
#include "Mesh.h"

namespace spark
{
struct RenderingRequest
{
    spark::ShaderType shaderType = spark::ShaderType::DEFAULT_SHADER;
    std::shared_ptr<GameObject> gameObject = nullptr;
    std::shared_ptr<Mesh> mesh = nullptr;
};
}  // namespace spark
