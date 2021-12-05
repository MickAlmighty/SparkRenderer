#pragma once

#include <filesystem>
#include <memory>

#include "AnimationData.hpp"
#include "AnimationPlayer.hpp"

namespace spark
{
class GameObject;

class AnimationCreator
{
    public:
    void setGameObject(const std::shared_ptr<GameObject>& go);
    void setTimeMarker(float tm);
    void startRecording();

    void addKeyFrame() const;
    void removeKeyFrame(float timeMarkerOfKeyFrame) const;

    void saveAnimation(const std::filesystem::path& path);

    void drawGui();

    private:
    std::weak_ptr<GameObject> gameObject{};
    std::shared_ptr<resources::AnimationData> animation = std::make_shared<resources::AnimationData>();
    AnimationPlayer animationPlayer;
    float timeMarker{0.0f};
};
}
