#include "AnimationCreator.hpp"

#include "Clock.h"
#include "GUI/ImGui/imgui.h"
#include "GUI/ImGui/imgui_custom_widgets.h"

#include "GameObject.h"
#include "JsonSerializer.h"
#include "SparkGui.h"
#include "glm/gtx/string_cast.hpp"

namespace spark
{
void AnimationCreator::setGameObject(const std::shared_ptr<GameObject>& go)
{
    gameObject = go;
}

void AnimationCreator::setTimeMarker(float tm)
{
    if(tm < 0.0f)
        tm = 0.0f;
    timeMarker = tm;
}

void AnimationCreator::startRecording()
{
    animation->clear();
    timeMarker = 0.0f;
    addKeyFrame();
}

void AnimationCreator::addKeyFrame() const
{
    animation->addKeyFrame(gameObject.lock(), timeMarker);
}

void AnimationCreator::removeKeyFrame(float timeMarkerOfKeyFrame) const
{
    animation->removeFrame(timeMarker);
}

void AnimationCreator::saveAnimation(const std::filesystem::path& path)
{
    JsonSerializer().save(animation, path);
}

void AnimationCreator::drawGui()
{
    if(ImGui::BeginDragDropTarget())
    {
        if(const ImGuiPayload* payload = ImGui::AcceptDragDropPayload("GAME_OBJECT"); payload)
        {
            const auto gameObjectWeak = static_cast<std::weak_ptr<GameObject>*>(payload->Data);
            gameObject = gameObjectWeak->lock();
            SPARK_INFO("shared_ptr use count {}", gameObject.use_count());
        }
        ImGui::EndDragDropTarget();
    }

    if(!gameObject.lock())
        ImGui::Text("GameObject");
    else
        ImGui::Text(gameObject.lock()->getName().c_str());

    if(!animation->empty() && gameObject.lock())
    {
        if(const auto animationPath = SparkGui::getRelativePathToSaveAnimationByFilePicker(); !animationPath.empty())
        {
            saveAnimation(animationPath);
        }

        ImGui::DragFloat("TimeMarker", &timeMarker);
        ImGui::SameLine();
        if(ImGui::Button("Add KeyFrame"))
        {
            addKeyFrame();
        }

        static bool isLooped{false};
        if(ImGui::Button("Play"))
        {
            animationPlayer.play(animation, 0, isLooped);
        }
        ImGui::SameLine();
        ImGui::Checkbox("Loop", &isLooped);

        if(ImGui::Button("Pause"))
        {
            animationPlayer.pause();
        }
        if(ImGui::Button("Stop"))
        {
            animationPlayer.stop(gameObject.lock());
        }

        if(animationPlayer.isPlayingAnimation())
        {
            animationPlayer.update(gameObject.lock(), Clock::getDeltaTime());
        }
    }
    else if(animation->empty() && gameObject.lock())
    {
        if(ImGui::Button("startRecording"))
        {
            startRecording();
        }
    }

    if(gameObject.lock())
    {
        ImGui::Separator();

        std::vector<float> timeMarkers;
        timeMarkers.reserve(animation->positionKeyFrames.size());

        for(auto& [timeMarker, value] : animation->positionKeyFrames)
        {
            timeMarkers.push_back(timeMarker);
        }

        for(auto tm : timeMarkers)
        {
            ImGui::BeginGroupPanel("KeyFrame");
            ImGui::Text("%f", tm);
            ImGui::SameLine();
            ImGui::TextColored({1.0f, 0.0f, 0.0f, 1.0f}, "P: %s", glm::to_string(animation->positionKeyFrames.at(tm)).c_str());
            ImGui::SameLine();
            ImGui::TextColored({1.0f, 0.0f, 1.0f, 1.0f}, "S: %s", glm::to_string(animation->scaleKeyFrames.at(tm)).c_str());
            ImGui::SameLine();
            ImGui::TextColored({0.0f, 0.0f, 1.0f, 1.0f}, "R: %s", glm::to_string(animation->rotationKeyFrames.at(tm)).c_str());
            ImGui::EndGroupPanel();
        }
    }
}
}  // namespace spark
