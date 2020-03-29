#include "ModelMesh.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>

#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "ResourceLibrary.h"

namespace spark
{
ModelMesh::ModelMesh() : Component("ModelMesh") {}

void ModelMesh::setModel(const std::shared_ptr<resources::Model>& model_)
{
    modelPath = model_->id.getFullPath().string();
    model = model_;
}

void ModelMesh::update()
{
    const glm::mat4 worldMatrix = getGameObject()->transform.world.getMatrix();
    for(const auto& mesh : model->getMeshes())
    {
        mesh->addToRenderQueue(worldMatrix);
    }
}

void ModelMesh::fixedUpdate() {}

void ModelMesh::drawGUI()
{
    for(const auto& mesh : model->getMeshes())
    {
        ImGui::Text("Vertices:");
        ImGui::SameLine();
        ImGui::Text(std::to_string(mesh->verticesCount).c_str());
        ImGui::Text("Indices:");
        ImGui::SameLine();
        ImGui::Text(std::to_string(mesh->indices.size()).c_str());
        ImGui::Text("Textures:");
        ImGui::SameLine();
        ImGui::Text(std::to_string(mesh->textures.size()).c_str());
        ImGui::Text("resources::Shader enum:");
        ImGui::SameLine();
        ImGui::Text(std::to_string(static_cast<int>(mesh->shaderType)).c_str());
        SparkGui::getShader();
        SparkGui::getTexture();
    }

    if(model->getMeshes().empty())
    {
        //#todo implement acquiring model from SparkGui
        // const auto model = SparkGui::getMeshes();
        // setModel(model);
    }

    removeComponentGUI<ModelMesh>();
}

std::string ModelMesh::getModelPath() const
{
    return modelPath;
}

void ModelMesh::setModelPath(const std::string modelPath)
{
    setModel(Spark::getResourceLibrary()->getResourceByPathWithOptLoad<resources::Model>(modelPath));
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::ModelMesh>("ModelMesh")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("modelPath", &spark::ModelMesh::getModelPath, &spark::ModelMesh::setModelPath, rttr::registration::public_access);
}