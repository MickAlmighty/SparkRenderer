#pragma once

#include <rttr/registration_friend>
#include <rttr/registration>
#include <glm/glm.hpp>

namespace spark
{
class WorldTransform final
{
    public:
    WorldTransform(glm::mat4 mat);
    ~WorldTransform() = default;

    glm::mat4 getMatrix() const;
    glm::vec3 getPosition() const;

    void setMatrix(glm::mat4 mat);
    void setPosition(glm::vec3 position);
    void setPosition(float x, float y, float z);
    void translate(glm::vec3 translation);
    void translate(float x, float y, float z);
    void setRotationRadians(glm::vec3& radians);
    void setRotationRadians(float x, float y, float z);
    void setRotationDegrees(glm::vec3& degrees);
    void setRotationDegrees(float x, float y, float z);
    void setScale(glm::vec3 scale);

    WorldTransform() = default;

    private:
    glm::mat4 modelMatrix{glm::mat4(1.0f)};
    bool dirty{true};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

}  // namespace spark