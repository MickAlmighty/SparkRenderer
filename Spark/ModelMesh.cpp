#include "ModelMesh.h"

#include <iostream>

#include <GUI/ImGui/imgui.h>

#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "RenderingRequest.h"
#include "ResourceLibrary.h"
#include "EngineSystems/SparkRenderer.h"

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

    if(model != nullptr)
    {
        for(const auto& mesh : model->getMeshes())
        {
            RenderingRequest request{};
            request.shaderType = mesh->shaderType;
            request.gameObject = getGameObject();
            request.mesh = mesh;
            request.model = getGameObject()->transform.world.getMatrix();

            SparkRenderer::getInstance()->addRenderingRequest(request);
        }
    }
}

void ModelMesh::fixedUpdate() {}

void ModelMesh::drawGUI()
{
    std::vector<std::shared_ptr<Mesh>> meshes{};

    if(model != nullptr)
    {
        meshes = model->getMeshes();
    }

    for(const auto& mesh : meshes)
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
    }

    if(model != nullptr)
    {
        if(model->getMeshes().empty())
        {
            const auto model = SparkGui::getModel();
            setModel(model);
        }
    }

    if(model == nullptr)
    {
        const auto model_ = SparkGui::getModel();
        if (model_ != nullptr)
        {
            setModel(model_);
        }
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