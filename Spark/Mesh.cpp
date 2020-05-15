#include "Mesh.h"

#include <iostream>

#include "EngineSystems/SparkRenderer.h"
#include "Shader.h"
#include "Logging.h"
#include "Texture.h"
#include "Timer.h"

namespace spark
{
Mesh::Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned>& indices,
           std::map<TextureTarget, std::shared_ptr<resources::Texture>>& meshTextures, std::string&& newName_, ShaderType shaderType)
{
    this->indices = std::move(indices);
    this->textures = std::move(meshTextures);
    this->shaderType = shaderType;

    for(auto& verticesAttribute : verticesAttributes)
    {
        attributesAndVbos.emplace_back(verticesAttribute, 0);
    }
    // Mesh::gpuLoad();
}

Mesh::Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned>& indices, std::string&& newName_, ShaderType shaderType)
{
    this->indices = std::move(indices);
    this->shaderType = shaderType;

    for(auto& verticesAttribute : verticesAttributes)
    {
        attributesAndVbos.emplace_back(verticesAttribute, 0);
    }
}

void Mesh::draw(std::shared_ptr<resources::Shader>& shader, glm::mat4 model)
{
    shader->setMat4("model", model);

    GLuint texturesToBind[5]{0,0,0,0,0};

    for(const auto& [textureTarget, texture] : textures)
    {
        texturesToBind[static_cast<GLuint>(textureTarget) - 1] = texture->getID();
    }

    if(!textures.empty())
        glBindTextures(1, 5, texturesToBind);

    glBindVertexArray(vao);

    if(!indices.empty())
    {
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, 0);
    }
    else
    {
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verticesCount));
    }

    glBindVertexArray(0);

    if(!textures.empty())
        glBindTextures(static_cast<GLuint>(TextureTarget::DIFFUSE_TARGET), 4, nullptr);
}

void Mesh::cleanup()
{
    gpuUnload();
    SPARK_TRACE("Mesh deleted!");
}

bool Mesh::gpuLoad()
{
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    std::vector<GLuint> bufferIDs;
    bufferIDs.resize(attributesAndVbos.size());

    if(!attributesAndVbos.empty())
    {
        glCreateBuffers(static_cast<GLsizei>(attributesAndVbos.size()), bufferIDs.data());
        verticesCount = static_cast<unsigned int>(attributesAndVbos[0].first.bytes.size()) / attributesAndVbos[0].first.stride;
    }

    unsigned int bufferIndex = 0;
    for(auto& [attribute, vboId] : attributesAndVbos)
    {
        glBindBuffer(GL_ARRAY_BUFFER, bufferIDs[bufferIndex]);
        glBufferData(GL_ARRAY_BUFFER, attribute.bytes.size(), reinterpret_cast<const void*>(attribute.bytes.data()), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(attribute.location);
        glVertexAttribPointer(attribute.location, attribute.components, GL_FLOAT, GL_FALSE, attribute.stride, reinterpret_cast<const void*>(0));

        vboId = bufferIDs[bufferIndex];

        ++bufferIndex;
    }

    if(!indices.empty())
    {
        glCreateBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), reinterpret_cast<const void*>(indices.data()), GL_DYNAMIC_DRAW);
    }

    glBindVertexArray(0);

    return true;
}

bool Mesh::gpuUnload()
{
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    for(auto& [attribute, vbo] : attributesAndVbos)
    {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }

    return true;
}
}  // namespace spark
