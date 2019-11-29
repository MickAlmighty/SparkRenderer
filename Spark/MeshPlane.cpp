#include "MeshPlane.h"

#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SparkRenderer.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"

namespace spark {

void MeshPlane::setup()
{
	vertices = {
		{{-1.0f, 1.0f, 0.0f},	{0.0f, 1.0f}},
		{{1.0f, 1.0f, 0.0f},	{1.0f, 1.0f }},
		{{1.0f, -1.0f, 0.0f},	{1.0f, 0.0f}},
		{{-1.0f, -1.0f, 0.0f},	{0.0f, 0.0f}}
	};

	indices = { 0,1,2, 2,3,0 };

	glCreateVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glCreateBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(QuadVertex), vertices.data(), GL_STATIC_DRAW);

	glCreateBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, pos)));

	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(QuadVertex), reinterpret_cast<void*>(offsetof(QuadVertex, texCoords)));

	glBindVertexArray(0);
}


MeshPlane::MeshPlane() : Component("MeshPlane") {
    setup();
}

MeshPlane::MeshPlane(std::string&& name) : Component(std::move(name)) {
    setup();
}

MeshPlane::~MeshPlane()
{
	glDeleteBuffers(1, &vbo);
	glDeleteBuffers(1, &ebo);
	glDeleteVertexArrays(1, &vao);
}

void MeshPlane::addToRenderQueue() const
{
	glm::mat4 model = getGameObject()->transform.world.getMatrix();
	auto f = [this, model](std::shared_ptr<Shader>& shader)
	{
		draw(shader, model);
	};
	SparkRenderer::getInstance()->renderQueue[shaderType].push_back(f);
}

void MeshPlane::draw(std::shared_ptr<Shader>& shader, glm::mat4 model) const
{
	shader->setMat4("model", model);

	for (auto& texture_it : textures)
	{
		glBindTextureUnit(static_cast<GLuint>(texture_it.first), texture_it.second.ID);
	}

	glBindVertexArray(vao);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()), GL_UNSIGNED_INT, nullptr);
	glBindVertexArray(0);

	glBindTextures(static_cast<GLuint>(TextureTarget::DIFFUSE_TARGET), static_cast<GLsizei>(textures.size()), nullptr);
}

void MeshPlane::setTexture(TextureTarget target, Texture tex)
{
	textures[target] = tex;
}

void MeshPlane::update()
{
	addToRenderQueue();
}

void MeshPlane::fixedUpdate()
{

}

void MeshPlane::drawGUI()
{
	ImGui::Text("Vertices:"); ImGui::SameLine(); ImGui::Text(std::to_string(vertices.size()).c_str());
	ImGui::Text("Indices:"); ImGui::SameLine(); ImGui::Text(std::to_string(indices.size()).c_str());
	ImGui::Text("Textures:"); ImGui::SameLine(); ImGui::Text(std::to_string(textures.size()).c_str());
	ImGui::Text("Shader enum:"); ImGui::SameLine(); ImGui::Text(std::to_string(static_cast<int>(shaderType)).c_str());
	ImGui::Separator();
	//ImGui::com
	static int mode = static_cast<int>(TextureTarget::DIFFUSE_TARGET);
	if (ImGui::RadioButton("Diffuse", mode == static_cast<int>(TextureTarget::DIFFUSE_TARGET))) { mode = static_cast<int>(TextureTarget::DIFFUSE_TARGET); } ImGui::SameLine();
	if (ImGui::RadioButton("Normal", mode == static_cast<int>(TextureTarget::NORMAL_TARGET))) { mode = static_cast<int>(TextureTarget::NORMAL_TARGET); }

	std::string name = "texture: " + std::to_string(textures[static_cast<TextureTarget>(mode)].ID);
	ImGui::Text(name.c_str());
	const auto optionalResult = SparkGui::getDraggedObject<Texture>("TEXTURE");
	if (optionalResult)
	{
		textures[static_cast<TextureTarget>(mode)] = optionalResult.value();
	}

	removeComponentGUI<MeshPlane>();
}
}

RTTR_REGISTRATION{
    rttr::registration::class_<spark::MeshPlane>("MeshPlane")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    .property("shaderType", &spark::MeshPlane::shaderType)
    .property("textures", &spark::MeshPlane::textures);
}