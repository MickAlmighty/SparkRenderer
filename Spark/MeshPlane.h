#ifndef MESH_PLANE_H
#define MESH_PLANE_H

#include "Component.h"
#include "Enums.h"
#include "Structs.h"

#include <vector>

namespace spark
{
class Shader;
class MeshPlane final : public Component
{
    public:
    MeshPlane();
    explicit MeshPlane(std::string&& name);
    ~MeshPlane();
    MeshPlane(const MeshPlane&) = delete;
    MeshPlane(const MeshPlane&&) = delete;
    MeshPlane& operator=(const MeshPlane&) = delete;
    MeshPlane& operator=(const MeshPlane&&) = delete;

    void setup();
    void addToRenderQueue() const;
    void draw(std::shared_ptr<Shader>& shader, glm::mat4 model) const;
    void setTexture(TextureTarget target, Texture tex);
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    private:
    GLuint vao{0}, vbo{0}, ebo{0};
    std::vector<QuadVertex> vertices;
    std::vector<unsigned int> indices;
    std::map<TextureTarget, Texture> textures;
    ShaderType shaderType{ShaderType::DEFAULT_SHADER};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component);
};

}  // namespace spark
#endif