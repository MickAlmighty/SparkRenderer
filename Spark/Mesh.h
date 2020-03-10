#ifndef MESH_H
#define MESH_H

#include <vector>
#include <map>

#include "Enums.h"
#include "Structs.h"

namespace spark
{
class Shader;
class Mesh : public std::enable_shared_from_this<Mesh>
{
    public:
    ShaderType shaderType = ShaderType::DEFAULT_SHADER;
    std::map<VertexShaderAttribute, GLuint> attributesAndVbos;
    unsigned int verticesCount{0};
    std::vector<unsigned int> indices;
    std::map<TextureTarget, Texture> textures;

    Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned int>& indices, std::map<TextureTarget, Texture>& meshTextures,
         std::string&& newName_ = "Mesh", ShaderType shaderType = ShaderType::DEFAULT_SHADER);
    ~Mesh() = default;

    void setup(std::vector<VertexShaderAttribute>& verticesAttributes);
    void addToRenderQueue(glm::mat4 model);
    void draw(std::shared_ptr<Shader>& shader, glm::mat4 model);
    void cleanup();

    private:
    GLuint vao{};
    GLuint vbo{};
    GLuint ebo{};
};

}  // namespace spark
#endif