#include "ModelMesh.h"

#include <iostream>

#include "GUI/ImGui/imgui.h"

#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "Logging.h"
#include "Mesh.h"
#include "Model.h"
#include "renderers/RenderingRequest.h"
#include "ResourceLibrary.h"
#include "Spark.h"
#include "EngineSystems/SparkRenderer.h"

namespace spark
{
ModelMesh::ModelMesh() : Component("ModelMesh") {}

void ModelMesh::setModel(const std::shared_ptr<resources::Model>& model_)
{
    modelPath = model_->getPath().string();
    model = model_;
}

void ModelMesh::update()
{
    const glm::mat4 worldMatrix = getGameObject()->transform.world.getMatrix();

    if(model != nullptr)
    {
        for(const auto& mesh : model->getMeshes())
        {
            renderers::RenderingRequest request{};
            request.shaderType = mesh->shaderType;
            request.gameObject = getGameObject();
            request.mesh = mesh;
            request.model = getGameObject()->transform.world.getMatrix();

            Spark::get().getRenderer().addRenderingRequest(request);
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

    const auto modelOpt = SparkGui::selectModelByFilePicker();
    if(modelOpt)
    {
        setModel(modelOpt.value());
    }

    removeComponentGUI<ModelMesh>();
}

std::string ModelMesh::getModelPath() const
{
    return modelPath;
}

void ModelMesh::setModelPath(const std::string modelPath)
{
    if(const auto modelPtr = Spark::get().getResourceLibrary().getResourceByRelativePath<resources::Model>(modelPath); modelPtr)
        setModel(modelPtr);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::ModelMesh>("ModelMesh")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("modelPath", &spark::ModelMesh::getModelPath, &spark::ModelMesh::setModelPath, rttr::registration::public_access);
}