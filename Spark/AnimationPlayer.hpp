#pragma once
#include <memory>

#include <rttr/registration_friend>

namespace spark
{
namespace resources {
    class AnimationData;
}

class GameObject;

class AnimationPlayer
{
    public:
    void update(const std::shared_ptr<GameObject>& gameObject, float deltaTime);

    bool isPlayingAnimation() const;
    
    void play(const std::shared_ptr<resources::AnimationData>& animData, float time = 0.0f, bool isLooped_ = false);
    void play(bool isLooped_ = false);
    void play(float time = 0.0f, bool isLooped_ = false);
    void pause();
    void stop(const std::shared_ptr<GameObject>& gameObject);

    void setAnimationData(const std::shared_ptr<const resources::AnimationData>& animData);

    void setTimeMarker(float tm);
    float getTimeMarker() const;
    void setLooped(bool looped);
    bool isAnimationLooped() const;

    private:
    bool interpolate(const std::shared_ptr<GameObject>& gameObject, float timePoint) const;

    bool isPlaying{false};
    bool isLooped{false};
    float timeMarker{0.0f};
    std::shared_ptr<const resources::AnimationData> animationData{nullptr};
    RTTR_REGISTRATION_FRIEND
};
}  // namespace spark