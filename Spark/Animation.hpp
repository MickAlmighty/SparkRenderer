#pragma once
#include "AnimationPlayer.hpp"
#include "Component.h"

namespace spark
{
class Animation : public Component
{
    public:
    void update() override;
    ~Animation() override = default;

    void play(bool isLooped = false);
    void pause();
    void stop();

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