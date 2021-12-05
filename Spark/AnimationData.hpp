#pragma once

#include <map>

#include <glm/vec3.hpp>
#include <glm/gtc/quaternion.hpp>

#include "Resource.h"

namespace spark
{
class GameObject;
}

namespace spark::resources
{
class AnimationData : public resourceManagement::Resource
{
    public:
    AnimationData();
    AnimationData(const std::filesystem::path& path_);

    void addKeyFrame(const std::shared_ptr<const GameObject>& gameObject, float timeMarker);
    void removeFrame(float timeMarker);

    bool interpolate(const std::shared_ptr<GameObject>& gameObject, float timePoint) const;

    bool empty() const;
    void clear();

    float getLastTimePoint() const;

    std::map<float, glm::vec3> positionKeyFrames;
    std::map<float, glm::vec3> scaleKeyFrames;
    std::map<float, glm::quat> rotationKeyFrames;
};
}  // namespace spark::resources