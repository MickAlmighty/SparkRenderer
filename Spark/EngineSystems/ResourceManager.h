#pragma once

#include <map>

#include "Structs.h"
#include "Enums.h"
#include "Shader.h"

namespace spark
{
    class Mesh;
    namespace resources
    {
        class Shader;
    }
}

namespace spark::deprecated
{

class ResourceManager
{
    public:
    static ResourceManager* getInstance();

    void addTexture(Texture tex);
    void addCubemapTexturePath(const std::string& path);
    Texture findTexture(const std::string&& path) const;
    std::vector<std::shared_ptr<Mesh>> findModelMeshes(const std::string& path) const;
    GLuint getTextureId(const std::string& path) const;
    std::vector<std::string> getPathsToModels() const;
    std::shared_ptr<resources::Shader> getShader(const ShaderType& type) const;
    std::shared_ptr<resources::Shader> getShader(const std::string& name) const;
    std::vector<std::string> getShaderNames() const;
    std::vector<Texture> getTextures() const;
    const std::vector<std::string>& getCubemapTexturePaths() const;
    void drawGui();
    void loadResources();
    void cleanup();

    private:
    std::vector<Texture> textures;
    std::map<std::string, std::vector<std::shared_ptr<Mesh>>> models;
    std::map<ShaderType, std::shared_ptr<resources::Shader>> shaders;
    std::vector<std::string> cubemapTexturePaths;

    ResourceManager() = default;
    ~ResourceManager() = default;
};
}