#pragma once

#include <vector>
#include <map>
#include <memory>
#include <string>

#include <glm/glm.hpp>

#include "Enums.h"
#include "glad_glfw3.h"
#include "VertexAttribute.hpp"

namespace spark
{
namespace resources
{
    class Shader;
    class Texture;
}  // namespace resources

class Mesh
{
    public:
    Mesh(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned int>& indices,
         const std::map<TextureTarget, std::shared_ptr<resources::Texture>>& meshTextures, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::PBR);
    Mesh(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned int>& indices, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::PBR);
    ~Mesh();

    void draw(std::shared_ptr<resources::Shader>& shader, glm::mat4 model);

    ShaderType shaderType = ShaderType::PBR;
    std::vector<std::pair<AttributeDescriptor, GLuint>> descriptorAndVboPairs;
    unsigned int verticesCount{0};
    unsigned int indicesCount{0};
    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;

    private:
    void load(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned>& indices);
    GLuint vao{};
    GLuint vbo{};
    GLuint ebo{};
};

}  // namespace spark