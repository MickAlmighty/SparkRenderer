#include "Scene.h"

#include "Camera.hpp"
#include "CameraManager.hpp"
#include "CommonUtils.h"
#include "EditorCamera.hpp"
#include "GameObject.h"
#include "GUI/ImGui/imgui.h"
#include "HID/HID.h"
#include "Logging.h"
#include "ResourceLoader.h"
#include "Spark.h"
#include "renderers/ClusterBasedForwardPlusRenderer.hpp"

namespace spark
{
Scene::Scene() : Resource("NewScene.scene")
{
    root = std::shared_ptr<GameObject>(new GameObject("Root"));
    root->setScene(this);
    editorCamera = std::make_shared<EditorCamera>(glm::vec3(0, 0, 5));
    cameraManager = std::make_shared<CameraManager>();
}

Scene::~Scene()
{
    root = nullptr;
    SPARK_TRACE("Scene destroyed!");
}

void Scene::update()
{
    clearRenderQueues();

    editorCamera->update();
    root->update();
    lightManager->updateLightBuffers();
}

std::shared_ptr<GameObject> Scene::spawnGameObject(std::string&& name)
{
    auto gameObject = std::shared_ptr<GameObject>(new GameObject(std::move(name)));
    gameObject->setScene(this);
    root->addChild(gameObject);
    return gameObject;
}

void Scene::renderGameThroughMainCamera()
{
    const auto camera = cameraManager->getMainCamera();
    if(!camera)
        return;

    PUSH_DEBUG_GROUP(RENDER_SCENE_MAIN_CAMERA_VIEW)

    const unsigned int width = 640, height = 360;
    if(!renderer)
    {
        renderer = std::make_unique<renderers::ClusterBasedForwardPlusRenderer>(width, height);
    }

    const auto lastW = Spark::get().getRenderingContext().width;
    const auto lastH = Spark::get().getRenderingContext().height;

    PUSH_DEBUG_GROUP(CLEAR_FBO_FOR_MAIN_CAMERA_VIEW)
    const glm::vec2 scale = glm::vec2(lastW / 1280.0f, lastH / 720.0f);
    const unsigned int scissorWidth = 160.0f * scale.x, scissorHeight = 90.0f * scale.y;
    const auto scissorX = lastW / 2 - scissorWidth / 2 - 5;
    const auto scissorY = lastH - (scissorHeight + 25);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glEnable(GL_SCISSOR_TEST);
    glScissor(scissorX, scissorY, scissorWidth, scissorHeight);
    glViewport(0, 0, lastW, lastH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glScissor(0, 0, lastW, lastH);
    glDisable(GL_SCISSOR_TEST);
    POP_DEBUG_GROUP()

    Spark::get().getRenderingContext().width = width;
    Spark::get().getRenderingContext().height = height;

    camera->update();
    renderer->render(shared_from_this(), camera, scissorX + 5, scissorY + 5, scissorWidth - 10, scissorHeight - 10);

    Spark::get().getRenderingContext().width = lastW;
    Spark::get().getRenderingContext().height = lastH;
    POP_DEBUG_GROUP()
}

void Scene::drawGUI()
{
    drawSceneGraph();
    renderGameThroughMainCamera();

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
    ImGui::Selectable(node->getName().c_str(), opened, 0, {node->getName().length() * 7.1f, 0.0f});

    if(ImGui::BeginDragDropSource(ImGuiDragDropFlags_None) && !isRootNode)
    {
        std::weak_ptr gameObjectWeak = node;
        ImGui::SetDragDropPayload("GAME_OBJECT", &gameObjectWeak,
                                  sizeof(std::weak_ptr<GameObject>));  // Set payload to carry the index of our item (could be anything)
        ImGui::Text("Getting reference to %s %d", node->getName().c_str(), node.use_count());
        ImGui::EndDragDropSource();
    }

    ImGui::OpenPopupOnItemClick("GraphNode_operations", 1);

    if(ImGui::BeginPopup("GraphNode_operations"))
    {
        ImGui::Text("Popup Example");

        if(ImGui::Button("Add child"))
        {
            node->addChild(spawnGameObject());
            ImGui::CloseCurrentPopup();
        }

        if(!isRootNode)
        {
            if(ImGui::Button("Delete"))
            {
                node->getParent()->removeChild(node);
                if(getGameObjectToPreview() == node)
                {
                    setGameObjectToPreview(nullptr);
                }
                ImGui::CloseCurrentPopup();
                ImGui::EndPopup();
                ImGui::PopID();
                return;
            }
        }
        ImGui::EndPopup();
    }

    if(ImGui::IsItemHovered() && ImGui::IsMouseClicked(0) && !isRootNode)
    {
        setGameObjectToPreview(node);
        SPARK_DEBUG("GameObject {} clicked!", node->getName());
    }

    if(!node->getChildren().empty())
    {
        ImGui::SameLine();
        if(ImGui::TreeNode("Children"))
        {
            for(auto i = 0; i < node->getChildren().size(); ++i)
            //for(const auto& child : node->getChildren())
            {
                //drawTreeNode(child, false);
                drawTreeNode(node->getChildren()[i], false);
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

std::shared_ptr<CameraManager> Scene::getCameraManager() const
{
    return cameraManager;
}

std::string Scene::getName() const
{
    return getPath().filename().string();
}

const std::map<ShaderType, std::deque<renderers::RenderingRequest>>& Scene::getRenderingQueues() const
{
    return renderingQueues;
}

std::weak_ptr<PbrCubemapTexture> Scene::getSkyboxCubemap() const
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

std::shared_ptr<GameObject> Scene::getRoot() const
{
    return root;
}

void Scene::setRoot(std::shared_ptr<GameObject> r)
{
    root = std::move(r);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::Scene>("Scene")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("root", &spark::Scene::getRoot, &spark::Scene::setRoot)
        .property("camera", &spark::Scene::editorCamera);
}
