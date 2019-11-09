#include "ModelMesh.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "Mesh.h"
#include "Logging.h"

namespace spark {

ModelMesh::ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName) : Component(std::move(modelName))
{
	this->meshes = meshes;
}

void ModelMesh::setModel(std::pair<std::string, std::vector<Mesh>> model)
{
	modelPath = model.first;
	meshes = model.second;
}

void ModelMesh::update()
{
	glm::mat4 model = getGameObject()->transform.world.getMatrix();
	for (Mesh& mesh : meshes)
	{
		mesh.addToRenderQueue(model);
	}
}

void ModelMesh::fixedUpdate()
{
}

void ModelMesh::drawGUI()
{
	for (Mesh& mesh : meshes)
	{
		ImGui::Text("Vertices:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.vertices.size()).c_str());
		ImGui::Text("Indices:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.indices.size()).c_str());
		ImGui::Text("Textures:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.textures.size()).c_str());
		ImGui::Text("Shader enum:"); ImGui::SameLine(); ImGui::Text(std::to_string(static_cast<int>(mesh.shaderType)).c_str());
		SparkGui::getShader();
	}

	if (meshes.empty())
	{
		const auto model = SparkGui::getMeshes();
		modelPath = model.first;
		meshes = model.second;
	}

	removeComponentGUI<ModelMesh>();
}
}

RTTR_REGISTRATION{
    rttr::registration::class_<spark::ModelMesh>("ModelMesh")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    .property("modelPath", &spark::ModelMesh::modelPath);
}