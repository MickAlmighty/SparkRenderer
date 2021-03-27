#pragma once

#include <rttr/registration_friend>
#include <rttr/registration>
#include <glm/glm.hpp>

namespace spark
{
class LocalTransform final
{
    public:
    LocalTransform(glm::vec3 pos, glm::vec3 scale, glm::vec3 rotation);
    ~LocalTransform() = default;

    void drawGUI();
    glm::mat4 getMatrix();
    glm::vec3 getPosition() const;
    glm::vec3 getScale() const;
    glm::vec3 getRotationRadians() const;
    glm::vec3 getRotationDegrees() const;

    void setPosition(float x, float y, float z);
    void setPosition(glm::vec3 pos);
    void translate(glm::vec3 translation);
    void translate(float x, float y, float z);
    void setScale(float x, float y, float z);
    void setScale(glm::vec3 scale);
    void setRotationDegrees(float x, float y, float z);
    void setRotationDegrees(glm::vec3 rotationDegrees);
    void setRotationRadians(float x, float y, float z);
    void setRotationRadians(glm::vec3 rotationRadians);

    LocalTransform() = default;

    private:
    glm::vec3 position{glm::vec3(0.0f)};
    glm::vec3 rotationEuler{glm::vec3(0.0f)};
    glm::vec3 scale{glm::vec3(1.0f)};
    glm::mat4 modelMatrix{glm::mat4(1.0f)};
    bool dirty{true};
    glm::mat4 recreateMatrix() const;
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE();
};

}  // namespace spark