#pragma once

#include <glm/glm.hpp>

namespace spark
{
struct Vertex final
{
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 texCoords;
    glm::vec3 tangent;
    glm::vec3 bitangent;
};
}  // namespace spark