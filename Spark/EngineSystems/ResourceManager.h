#ifndef RESOURCE_MANAGER_H
#define RESOURCE_MANAGER_H

#include <map>

#include "Structs.h"
#include "Enums.h"

namespace spark
{
class Mesh;
class Shader;
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
    std::shared_ptr<Shader> getShader(const ShaderType& type) const;
    std::shared_ptr<Shader> getShader(const std::string& name) const;
    std::vector<std::string> getShaderNames() const;
    std::vector<Texture> getTextures() const;
    const std::vector<std::string>& getCubemapTexturePaths() const;
    void drawGui();
    void loadResources();
    void cleanup();

    private:
    std::vector<Texture> textures;
    std::map<std::string, std::vector<std::shared_ptr<Mesh>>> models;
    std::map<ShaderType, std::shared_ptr<Shader>> shaders;
    std::vector<std::string> cubemapTexturePaths;

    ResourceManager() = default;
    ~ResourceManager() = default;
};
}  // namespace spark
#endif