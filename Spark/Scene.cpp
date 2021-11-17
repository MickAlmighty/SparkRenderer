#include "Scene.h"

#include "Camera.h"
#include "GameObject.h"
#include "GUI/ImGui/imgui.h"
#include "HID/HID.h"
#include "JsonSerializer.h"
#include "Logging.h"
#include "ResourceLoader.h"

#include <iostream>

namespace spark
{
Scene::Scene() : Resource("NewScene.scene")
{
    init();
}

Scene::Scene(const std::filesystem::path& path_) : Resource(path_)
{
    init();
}

Scene::Scene(const std::filesystem::path& path_, const std::shared_ptr<Scene>&& scene_) : Resource(path_)
{
    camera = std::move(scene_->camera);
    root = std::move(scene_->root);
    lightManager = std::move(scene_->lightManager);
    skyboxCubemap = std::move(scene_->skyboxCubemap);
    root->setSceneRecursive(this);
}

Scene::~Scene()
{
    root = nullptr;
    SPARK_TRACE("Scene destroyed!");
}

void Scene::update()
{
    clearRenderQueues();

    camera->update();
    root->update();
    lightManager->updateLightBuffers();
}

std::shared_ptr<Camera> Scene::getCamera() const
{
    return camera;
}

std::shared_ptr<GameObject> Scene::getRoot() const
{
    return root;
}

void Scene::drawGUI()
{
    drawSceneGraph();
    const auto goToPreview = getGameObjectToPreview();
    if(goToPreview != nullptr && !isGameObjectPreviewOpened)
    {
        isGameObjectPreviewOpened = true;
    }
    else if(goToPreview == nullptr && isGameObjectPreviewOpened)
    {
        isGameObjectPreviewOpened = false;
    }

    if(isGameObjectPreviewOpened)
    {
        ImGuiIO& io = ImGui::GetIO();

        ImGui::SetNextWindowPos({io.DisplaySize.x - 5, 25}, ImGuiCond_Always, {1, 0});
        ImGui::SetNextWindowSizeConstraints(ImVec2(350, 20), ImVec2(350, io.DisplaySize.y - 50));

        if(ImGui::Begin("GameObject", &isGameObjectPreviewOpened,
                        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_AlwaysHorizontalScrollbar | ImGuiWindowFlags_NoCollapse))
        {
            goToPreview->drawGUI();
            ImGui::End();
        }

        if(!isGameObjectPreviewOpened)
        {
            setGameObjectToPreview(nullptr);
        }
    }
}

void Scene::init()
{
    root = std::make_shared<GameObject>("Root");
    root->setScene(this);
    camera = std::make_shared<Camera>(glm::vec3(0, 0, 5));
}

void Scene::drawSceneGraph()
{
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos({5, 25}, ImGuiCond_Always, {0, 0});
    if(ImGui::Begin("Scene", NULL, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_HorizontalScrollbar))
    {
        ImGui::Text(getName().c_str());
        ImGui::Separator();
        drawTreeNode(root, true);
    }
    ImGui::End();
}

void Scene::drawTreeNode(std::shared_ptr<GameObject> node, bool isRootNode)
{
    ImGui::PushID(node.get());
    bool opened = node == getGameObjectToPreview();
    ImGui::Selectable(node->name.c_str(), opened, 0, {node->name.length() * 7.1f, 0.0f});

    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None) && !isRootNode)
    {
        ImGui::SetDragDropPayload("OBJECT_DRAG_AND_DROP", &node,
                                  sizeof(std::shared_ptr<GameObject>));  // Set payload to carry the index of our item (could be anything)
        ImGui::Text("Getting reference to %s", node->name.c_str());
        ImGui::EndDragDropSource();
    }

    ImGui::OpenPopupOnItemClick("GraphNode_operations", 1);

    if(ImGui::BeginPopup("GraphNode_operations"))
    {
        ImGui::Text("Popup Example");

        if(ImGui::Button("Add child"))
        {
            node->addChild(std::make_shared<GameObject>());
            ImGui::CloseCurrentPopup();
        }

        if(!isRootNode)
        {
            if(ImGui::Button("Delete"))
            {
                node->getParent()->removeChild(node);
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::EndPopup();
    }

    if(ImGui::IsItemHovered() && ImGui::IsMouseClicked(0) && !isRootNode)
    {
        setGameObjectToPreview(node);
        SPARK_DEBUG("GameObject {} clicked!", node->name);
    }

    if(!node->children.empty())
    {
        ImGui::SameLine();
        if(ImGui::TreeNode("Children"))
        {
            for(size_t i = 0; i < node->children.size(); ++i)
            {
                drawTreeNode(node->children[i], false);
            }

            ImGui::TreePop();
        }
    }
    ImGui::PopID();
}

std::shared_ptr<GameObject> Scene::getGameObjectToPreview() const
{
    return gameObjectToPreview.lock();
}

std::string Scene::getName() const
{
    return getPath().filename().string();
}

const std::map<ShaderType, std::deque<renderers::RenderingRequest>>& Scene::getRenderingQueues() const
{
    return renderingQueues;
}

const std::weak_ptr<PbrCubemapTexture> Scene::getSkyboxCubemap() const
{
    return skyboxCubemap;
}

void Scene::addRenderingRequest(const renderers::RenderingRequest& request)
{
    renderingQueues[request.shaderType].push_back(request);
}

void Scene::setCubemap(const std::shared_ptr<PbrCubemapTexture>& cubemap)
{
    skyboxCubemap = cubemap;
}

void Scene::setGameObjectToPreview(const std::shared_ptr<GameObject> node)
{
    gameObjectToPreview = node;
}

void Scene::clearRenderQueues()
{
    for(auto& [shaderType, shaderRenderList] : renderingQueues)
    {
        shaderRenderList.clear();
    }
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Scene>("Scene")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("root", &spark::Scene::root)
        .property("camera", &spark::Scene::camera);
}
