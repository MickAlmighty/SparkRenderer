#include "AnimationPlayer.hpp"

#include <rttr/registration>

#include "AnimationData.hpp"

namespace spark
{
bool AnimationPlayer::isPlayingAnimation() const
{
    return isPlaying;
}

bool AnimationPlayer::isAnimationLooped() const
{
    return isLooped;
}

void AnimationPlayer::play(const std::shared_ptr<resources::AnimationData>& animData, float time, bool isLooped_)
{
    setAnimationData(animData);
    play(time, isLooped_);
}

void AnimationPlayer::play(bool isLooped_)
{
    play(0.0f, isLooped_);
}

void AnimationPlayer::play(float time, bool isLooped_)
{
    timeMarker = time;
    isLooped = isLooped_;
    isPlaying = true;
}

void AnimationPlayer::pause()
{
    isPlaying = false;
}

void AnimationPlayer::stop(const std::shared_ptr<GameObject>& gameObject)
{
    if(animationData)
    {
        isPlaying = false;
        timeMarker = 0.0f;
        interpolate(gameObject, timeMarker);
    }
}

void AnimationPlayer::setAnimationData(const std::shared_ptr<const resources::AnimationData>& animData)
{
    animationData = animData;
}

void AnimationPlayer::setLooped(bool looped)
{
    isLooped = looped;
}

void AnimationPlayer::update(const std::shared_ptr<GameObject>& gameObject, float deltaTime)
{
    if(!isPlaying || !animationData)
        return;

    timeMarker += deltaTime;

    if(const float lastTimePoint = animationData->getLastTimePoint(); timeMarker > lastTimePoint)
    {
        if(isLooped)
        {
            timeMarker = timeMarker - lastTimePoint;
        }
        else
        {
            timeMarker = lastTimePoint;
            isPlaying = false;
        }
    }

    if(!interpolate(gameObject, timeMarker))
    {
        timeMarker = 0.0f;
        if(!isLooped)
        {
            isPlaying = false;
        }
    }
}

bool AnimationPlayer::interpolate(const std::shared_ptr<GameObject>& gameObject, float timePoint) const
{
    if(animationData)
    {
        return animationData->interpolate(gameObject, timePoint);
    }
    return false;
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::AnimationPlayer>("AnimationPlayer")
        .constructor()(rttr::policy::ctor::as_object)
        .property("isPlaying", &spark::AnimationPlayer::isPlaying)
        .property("isLooped", &spark::AnimationPlayer::isLooped)
        .property("timeMarker", &spark::AnimationPlayer::timeMarker);
}