#pragma once

#include <vector>

#include <glm/vec3.hpp>

namespace spark
{
    class ShapeCreator
    {
    public:
        static std::vector<glm::vec3> createSphere(float radius, int precision, glm::vec3 centerPoint = glm::vec3(0.0f));
    private:
        static void createSphereSegment(std::vector<glm::vec3>* vertices, float angle, float radStep, float radius, int precision, glm::vec3 centerPoint);
        static void createRectangle(std::vector<glm::vec3>* vertices, const glm::vec3& tL, const glm::vec3& tR, const glm::vec3& dR, const glm::vec3& dL, glm::vec3 centerPoint);
        static void createTriangle(std::vector<glm::vec3>* vertices, const glm::vec3& up, const glm::vec3& right, const glm::vec3& left, glm::vec3 centerPoint);
    };
}