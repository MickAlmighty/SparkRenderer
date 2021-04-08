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

class Mesh : public std::enable_shared_from_this<Mesh>
{
    public:
    ShaderType shaderType = ShaderType::DEFAULT_SHADER;
    std::vector<std::pair<VertexShaderAttribute, GLuint>> attributesAndVbos;
    unsigned int verticesCount{0};
    std::vector<unsigned int> indices;
    std::map<TextureTarget, std::shared_ptr<resources::Texture>> textures;

    Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned int>& indices,
         std::map<TextureTarget, std::shared_ptr<resources::Texture>>& meshTextures, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::DEFAULT_SHADER);
    Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned int>& indices, std::string&& newName_ = "Mesh",
         ShaderType shaderType = ShaderType::DEFAULT_SHADER);
    ~Mesh();

    void draw(std::shared_ptr<resources::Shader>& shader, glm::mat4 model);

    private:
    void load();
    GLuint vao{};
    GLuint vbo{};
    GLuint ebo{};
};

}  // namespace spark