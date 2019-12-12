#include "ModelMesh.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>

#include "EngineSystems/ResourceManager.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "Mesh.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark {

	ModelMesh::ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName) : Component(modelName)
	{
		this->meshes = meshes;
	}

	void ModelMesh::setModel(const std::pair<std::string, std::vector<Mesh>>& model)
	{
		modelPath = model.first;
		meshes = model.second;
	}

	ModelMesh::~ModelMesh()
	{
#ifdef DEBUG
		std::cout << "ModelMesh deleted!" << std::endl;
#endif
	}

	void ModelMesh::update()
	{
		const glm::mat4 model = getGameObject()->transform.world.getMatrix();
		if (instanced)
		{
			SparkRenderer::getInstance()->renderInstancedQueue[ShaderType::DEFAULT_INSTANCED_SHADER].push_back(shared_from_base<ModelMesh>());
		}
		else
		{
			for (Mesh& mesh : meshes)
			{
				mesh.addToRenderQueue(model);
			}
		}
	}

	void ModelMesh::fixedUpdate()
	{
	}

	void ModelMesh::drawGUI()
	{
		ImGui::Checkbox("Instanced", &instanced);
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

	SerializableType ModelMesh::getSerializableType()
	{
		return SerializableType::SModelMesh;
	}

	Json::Value ModelMesh::serialize()
	{
		Json::Value root;
		root["modelPath"] = modelPath;
		root["name"] = name;
		root["instanced"] = instanced;
		root["active"] = active;
		return root;
	}

	void ModelMesh::deserialize(Json::Value& root)
	{
		modelPath = root.get("modelPath", "").asString();
		name = root.get("name", "ModelMesh").asString();
		meshes = ResourceManager::getInstance()->findModelMeshes(modelPath);
		instanced = root.get("instanced", false).asBool();
		active = root.get("active", true).asBool();
	}
}
