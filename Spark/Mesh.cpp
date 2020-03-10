#include "Mesh.h"

#include <iostream>

#include "EngineSystems/SparkRenderer.h"
#include "Shader.h"
#include "Logging.h"

namespace spark
{
Mesh::Mesh(std::vector<VertexShaderAttribute>& verticesAttributes, std::vector<unsigned>& indices, std::map<TextureTarget, Texture>& meshTextures,
           std::string&& newName_, ShaderType shaderType)
{
    this->indices = std::move(indices);
    this->textures = std::move(meshTextures);
    this->shaderType = shaderType;

    setup(verticesAttributes);
}

void Mesh::setup(std::vector<VertexShaderAttribute>& verticesAttributes)
{
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    std::vector<GLuint> bufferIDs;
    bufferIDs.resize(verticesAttributes.size());

    if(!verticesAttributes.empty())
    {
        glCreateBuffers(static_cast<GLsizei>(verticesAttributes.size()), bufferIDs.data());
        verticesCount = static_cast<unsigned int>(verticesAttributes[0].bytes.size()) / verticesAttributes[0].stride;
    }

    unsigned int bufferIndex = 0;
    for(const auto& attribute : verticesAttributes)
    {
        glBindBuffer(GL_ARRAY_BUFFER, bufferIDs[bufferIndex]);
        glBufferData(GL_ARRAY_BUFFER, attribute.bytes.size(), reinterpret_cast<const void*>(attribute.bytes.data()), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(attribute.location);
        glVertexAttribPointer(attribute.location, attribute.components, GL_FLOAT, GL_FALSE, attribute.stride, reinterpret_cast<const void*>(0));

        attributesAndVbos.insert({attribute, bufferIDs[bufferIndex]});

        ++bufferIndex;
    }

    if(!indices.empty())
    {
        glCreateBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), reinterpret_cast<const void*>(indices.data()), GL_DYNAMIC_DRAW);
    }

    glBindVertexArray(0);
}

void Mesh::addToRenderQueue(glm::mat4 model)
{
    const auto thisPtr = shared_from_this();
    auto f = [thisPtr, model](std::shared_ptr<Shader>& shader) { thisPtr->draw(shader, model); };
    SparkRenderer::getInstance()->renderQueue[shaderType].push_back(f);
}

void Mesh::draw(std::shared_ptr<Shader>& shader, glm::mat4 model)
{
    shader->setMat4("model", model);

    for(auto& texture_it : textures)
    {
        glBindTextureUnit(static_cast<GLuint>(texture_it.first), texture_it.second.ID);
    }

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
        glBindTextures(static_cast<GLuint>(TextureTarget::DIFFUSE_TARGET), static_cast<GLsizei>(textures.size()), nullptr);
}

void Mesh::cleanup()
{
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    for(auto& [attribute, vbo] : attributesAndVbos)
    {
        glDeleteBuffers(1, &vbo);
    }

    attributesAndVbos.clear();
    SPARK_TRACE("Mesh deleted!");
}

}  // namespace spark
