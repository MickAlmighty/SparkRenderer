#pragma once
#include "AnimationPlayer.hpp"
#include "Component.h"

namespace spark
{
class Animation : public Component
{
    public:
    Animation() = default;
    ~Animation() override = default;

    void update() override;

    void play(bool isLooped = false);
    void pause();
    void stop();
    bool isPlaying() const;
    bool hasData() const;

    std::string getAnimationDataRelativePath() const;
    void setAnimationDataByRelativePath(std::string path);

    protected:
    void drawUIBody() override;

    private:
    AnimationPlayer animationPlayer{};
    std::shared_ptr<const resources::AnimationData> animationData;

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(Component)
};

}  // namespace spark