#pragma once

#include <vector>
#include <map>

#include "Enums.h"
#include "Structs.h"

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
    Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned int>& indices,
         std::map<TextureTarget, std::shared_ptr<resources::Texture>>& meshTextures, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::PBR);
    Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned int>& indices, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::PBR);
    ~Mesh();

    void draw(std::shared_ptr<resources::Shader>& shader, glm::mat4 model);

    ShaderType shaderType = ShaderType::PBR;
    std::vector<std::pair<AttributeDescriptor, GLuint>> descriptorAndVboPairs;
    unsigned int verticesCount{0};
    unsigned int indicesCount{0};
    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;

    private:
    void load(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned>& indices);
    GLuint vao{};
    GLuint vbo{};
    GLuint ebo{};
};

}  // namespace spark