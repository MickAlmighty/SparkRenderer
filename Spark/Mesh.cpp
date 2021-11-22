#include "Mesh.h"

#include <array>

#include "Shader.h"
#include "Logging.h"
#include "Texture.h"

namespace spark
{
Mesh::Mesh(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned>& indices,
           const std::map<TextureTarget, std::shared_ptr<resources::Texture>>& meshTextures, std::string&& newName_, ShaderType shaderType)
{
    this->textures = std::move(meshTextures);
    this->shaderType = shaderType;
    load(verticesAttributes, indices);
}

Mesh::Mesh(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned>& indices, std::string&& newName_, ShaderType shaderType)
{
    this->shaderType = shaderType;
    load(verticesAttributes, indices);
}

Mesh::~Mesh()
{
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ebo);
    glDeleteVertexArrays(1, &vao);

    for(auto& [descriptor, vbo] : descriptorAndVboPairs)
    {
        glDeleteBuffers(1, &vbo);
        vbo = 0;
    }
}

void Mesh::draw(std::shared_ptr<resources::Shader>& shader, glm::mat4 model)
{
    shader->setMat4("model", model);

    std::array<GLuint, 6> texturesToBind{0, 0, 0, 0, 0, 0};

    for(const auto& [textureTarget, texture] : textures)
    {
        texturesToBind[static_cast<GLuint>(textureTarget) - 1] = texture->getID();
    }

    if(!textures.empty())
        glBindTextures(1, static_cast<GLsizei>(texturesToBind.size()), texturesToBind.data());

    glBindVertexArray(vao);

    if(indicesCount != 0)
    {
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indicesCount), GL_UNSIGNED_INT, 0);
    }
    else
    {
        glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verticesCount));
    }

    glBindVertexArray(0);

    if(!textures.empty())
        glBindTextures(static_cast<GLuint>(TextureTarget::DIFFUSE_TARGET), static_cast<GLsizei>(texturesToBind.size()), nullptr);
}

void Mesh::load(const std::vector<VertexAttribute>& verticesAttributes, const std::vector<unsigned>& indices)
{
    glCreateVertexArrays(1, &vao);
    glBindVertexArray(vao);

    std::vector<GLuint> bufferIDs;
    bufferIDs.resize(verticesAttributes.size());
    descriptorAndVboPairs.reserve(verticesAttributes.size());

    if(!verticesAttributes.empty())
    {
        glCreateBuffers(static_cast<GLsizei>(verticesAttributes.size()), bufferIDs.data());
        verticesCount = static_cast<unsigned int>(verticesAttributes[0].bytes.size()) / verticesAttributes[0].descriptor.stride;
    }

    for(size_t i = 0; i < verticesAttributes.size(); ++i)
    {
        descriptorAndVboPairs.emplace_back(verticesAttributes[i].descriptor, 0);
        glBindBuffer(GL_ARRAY_BUFFER, bufferIDs[i]);
        glBufferData(GL_ARRAY_BUFFER, verticesAttributes[i].bytes.size(), reinterpret_cast<const void*>(verticesAttributes[i].bytes.data()),
                     GL_DYNAMIC_DRAW);

        const auto& descriptor = verticesAttributes[i].descriptor;
        glEnableVertexAttribArray(descriptor.location);
        glVertexAttribPointer(descriptor.location, descriptor.components, GL_FLOAT, GL_FALSE, descriptor.stride, reinterpret_cast<const void*>(0));

        descriptorAndVboPairs[i].second = bufferIDs[i];
    }

    if(!indices.empty())
    {
        indicesCount = indices.size();
        glCreateBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), reinterpret_cast<const void*>(indices.data()), GL_DYNAMIC_DRAW);
    }

    glBindVertexArray(0);
}
}  // namespace spark
