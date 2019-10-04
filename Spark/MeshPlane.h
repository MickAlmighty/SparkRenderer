#ifndef MESH_PLANE_H
#define MESH_PLANE_H

#include <vector>

#include "Component.h"
#include "Enums.h"
#include "Structs.h"

namespace spark {

class Shader;
class MeshPlane : public Component
{
public:
	MeshPlane(std::string&& newName = "MeshPlane");
	~MeshPlane();
	
	void setup();
	void addToRenderQueue() const;
	void draw(std::shared_ptr<Shader>& shader, glm::mat4 model) const;
	void setTexture(TextureTarget target, Texture tex);
	
	SerializableType getSerializableType() override;
	Json::Value serialize() override;
	void deserialize(Json::Value& root) override;
	void update() override;
	void fixedUpdate() override;
	void drawGUI() override;

private:
	GLuint vao{ 0 }, vbo{ 0 }, ebo{ 0 };
	std::vector<QuadVertex> vertices;
	std::vector<unsigned int> indices;
	std::map<TextureTarget, Texture> textures;
	ShaderType shaderType = ShaderType::DEFAULT_SHADER;
};

}
#endif