#include <ModelMesh.h>
#include <GUI/ImGui/imgui.h>
#include "GUI/SparkGui.h"

ModelMesh::ModelMesh(std::vector<Mesh>& meshes, std::string&& modelName) : Component(modelName)
{
	this->meshes = meshes;
}

ModelMesh::ModelMesh()
{
}

void ModelMesh::setMeshes(std::vector<Mesh>& meshes)
{
	this->meshes = meshes;
}

ModelMesh::~ModelMesh()
{
#ifdef DEBUG
	std::cout << "ModelMesh deleted!" << std::endl;
#endif
}

void ModelMesh::update()
{
	glm::mat4 model = getGameObject()->transform.world.getMatrix();
	for(Mesh& mesh: meshes)
	{
		mesh.addToRenderQueue(model);
	}
}

void ModelMesh::fixedUpdate()
{
}

void ModelMesh::drawGUI()
{
	ImGui::PushID(this);
	ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 5.0f);
	ImGui::SetNextWindowSizeConstraints(ImVec2(250, 0), ImVec2(FLT_MAX, 150)); // Width = 250, Height > 100
	ImGui::BeginChild("ModelMesh", { 0, 0 }, true, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_AlwaysAutoResize);
	if (ImGui::BeginMenuBar())
	{
		ImGui::Text("ModelMesh");
		ImGui::EndMenuBar();
	}
	for(Mesh& mesh: meshes)
	{
		ImGui::Text("Vertices:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.vertices.size()).c_str());
		ImGui::Text("Indices:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.indices.size()).c_str());
		ImGui::Text("Textures:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.textures.size()).c_str());
		ImGui::Text("Shader enum:"); ImGui::SameLine(); ImGui::Text(std::to_string(mesh.shaderType).c_str());
		ImGui::Separator();
	}

	if(meshes.empty())
	{
		meshes = SparkGui::getMeshes();
	}

	removeComponentGUI<ModelMesh>();

	ImGui::EndChild();
	ImGui::PopStyleVar();
	ImGui::PopID();
}
