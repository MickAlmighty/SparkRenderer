#include "AnimationData.hpp"

#include <optional>
#include <rttr/registration.h>

#include "GameObject.h"

namespace
{
glm::vec3 interpolate(glm::vec3 lhs, glm::vec3 rhs, float t)
{
    return glm::mix(lhs, rhs, t);
}

glm::quat interpolate(glm::quat lhs, glm::quat rhs, float t)
{
    return glm::slerp(lhs, rhs, t);
}

template<typename T>
std::optional<T> getInterpolatedValue(const std::map<float, T>& map, float timeMarker)
{
    const auto isBetweenKeyframes = [timeMarker](auto& it1, auto& it2) { return timeMarker >= it1.first && timeMarker <= it2.first; };

    if(const auto it = std::adjacent_find(map.begin(), map.end(), isBetweenKeyframes); it != map.end())
    {
        const auto& [kfTime1, value1] = *it;
        const auto& [kfTime2, value2] = *std::next(it);

        const float t = (timeMarker - kfTime1) / (kfTime2 - kfTime1);  // <0,1)

        return interpolate(value1, value2, t);
    }

    return std::nullopt;
}
}  // namespace

namespace spark::resources
{
AnimationData::AnimationData() {}
AnimationData::AnimationData(const std::filesystem::path& path_) : Resource(path_) {}

void AnimationData::addKeyFrame(const std::shared_ptr<const GameObject>& gameObject, float timeMarker)
{
    if(gameObject)
    {
        const auto& localTransform = gameObject->transform.local;

        positionKeyFrames.insert_or_assign(timeMarker, localTransform.getPosition());
        rotationKeyFrames.insert_or_assign(timeMarker, localTransform.getRotation());
        scaleKeyFrames.insert_or_assign(timeMarker, localTransform.getScale());
    }
}

void AnimationData::removeFrame(float timeMarker)
{
    positionKeyFrames.erase(timeMarker);
    scaleKeyFrames.erase(timeMarker);
    rotationKeyFrames.erase(timeMarker);
}

bool AnimationData::empty() const
{
    return positionKeyFrames.empty() && scaleKeyFrames.empty() && rotationKeyFrames.empty();
}

void AnimationData::clear()
{
    positionKeyFrames.clear();
    scaleKeyFrames.clear();
    rotationKeyFrames.clear();
}

float AnimationData::getLastTimePoint() const
{
    float lastTimePoint{0.0f};
    if(!positionKeyFrames.empty())
    {
        lastTimePoint = glm::max(positionKeyFrames.rbegin()->first, lastTimePoint);
    }
    if(!scaleKeyFrames.empty())
    {
        lastTimePoint = glm::max(scaleKeyFrames.rbegin()->first, lastTimePoint);
    }
    if(!rotationKeyFrames.empty())
    {
        lastTimePoint = glm::max(rotationKeyFrames.rbegin()->first, lastTimePoint);
    }

    return lastTimePoint;
}

bool AnimationData::interpolate(const std::shared_ptr<GameObject>& gameObject, float timePoint) const
{
    if(gameObject)
    {
        const auto positionOpt = getInterpolatedValue(positionKeyFrames, timePoint);
        const auto scaleOpt = getInterpolatedValue(scaleKeyFrames, timePoint);
        const auto rotationOpt = getInterpolatedValue(rotationKeyFrames, timePoint);

        bool isGoInterpolated{false};
        if(positionOpt)
        {
            gameObject->transform.local.setPosition(positionOpt.value());
            isGoInterpolated = true;
        }

        if(scaleOpt)
        {
            gameObject->transform.local.setScale(scaleOpt.value());
            isGoInterpolated = true;
        }

        if(rotationOpt)
        {
            gameObject->transform.local.setRotation(rotationOpt.value());
            isGoInterpolated = true;
        }

        return isGoInterpolated;
    }
    return false;
}
}  // namespace spark::resources

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::resources::AnimationData>("AnimationData")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("positionKeyFrames", &spark::resources::AnimationData::positionKeyFrames)
        .property("rotationKeyFrames", &spark::resources::AnimationData::rotationKeyFrames)
        .property("scaleKeyFrames", &spark::resources::AnimationData::scaleKeyFrames);
}