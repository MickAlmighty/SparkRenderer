#include "ShapeCreator.h"

#define _USE_MATH_DEFINES
#include <math.h>

#include <glm/gtx/rotate_vector.hpp>

namespace spark
{

	std::vector<glm::vec3> ShapeCreator::createSphere(float radius, int precision, glm::vec3 centerPoint)
	{
		if (radius < 0.0f)
			return {};

		if (precision < 3)
			return {};

		std::vector<glm::vec3> vertices;

		const float radStep = 2.0f * static_cast<float>(M_PI) / static_cast<float>(precision);
		float angle = 0.0f;

		for (int i = 0; i < precision; ++i)
		{
			createSphereSegment(&vertices, angle, i == precision - 1 ? 2.0f * static_cast<float>(M_PI) - angle : radStep, radius, precision, centerPoint);
			angle += radStep;
		}

		return vertices;
	}

	void ShapeCreator::createSphereSegment(std::vector<glm::vec3>* vertices, float angle, float radStep, float radius, int precision, glm::vec3 centerPoint)
	{
		std::vector<glm::vec3> circle(precision);

		float circleAngle = -M_PI / 2.0f;
		const float vertRadStep = M_PI / (static_cast<float>(precision) - 1.0f);

		for (int i = 0; i < precision; ++i) 
		{
			if (i == precision - 1) 
			{
				circle[i].x = 0.0f;
				circle[i].y = radius;
			}
			else if (i == 0) 
			{
				circle[i].x = 0.0f;
				circle[i].y = -radius;
			}
			else 
			{
				circle[i].x = radius * cos(circleAngle);
				circle[i].y = radius * sin(circleAngle);
			}
			circle[i].z = 0.0f;

			circleAngle += vertRadStep;
		}

		const glm::vec3 rotateAxis(0.0f, 1.0f, 0.0f);

		for (int i = 0; i < precision; i++)
		{
			circle[i] = rotate(circle[i], angle, rotateAxis);
		}

		std::vector<glm::vec3> circle2(precision);

		std::memcpy(circle2.data(), circle.data(), precision * sizeof(glm::vec3));

		for (int i = 0; i < precision; ++i) 
		{
			circle2[i] = rotate(circle2[i], radStep, rotateAxis);
		}

		for (int i = 0; i < precision - 1; ++i) 
		{
			if (i == 0) 
			{
				createTriangle(vertices, circle[i], circle[i + 1], circle2[i + 1], centerPoint);
			}
			else if (i == precision - 2) 
			{
				createTriangle(vertices, circle[i + 1], circle2[i], circle[i], centerPoint);
			}
			else 
			{
				createRectangle(vertices, circle[i], circle2[i], circle2[i + 1], circle[i + 1], centerPoint);
			}
		}
	}

    void ShapeCreator::createRectangle(std::vector<glm::vec3>* vertices, const glm::vec3& tL, const glm::vec3& tR, const glm::vec3& dR, const glm::vec3& dL, glm::vec3 centerPoint)
    {
		const glm::vec3 horizontal = dR - dL;
		const glm::vec3 vertical = tL - dL;
		//glm::vec3 normal = cross(vertical, horizontal);
		glm::vec3 output[4];

		/*for (int i = 0; i < 4; i++) {
			output[i].Normal = normal;
		}*/
		output[0] =  tL + centerPoint;
		output[1] = tR + centerPoint;
		output[2] = dR + centerPoint;
		output[3] = dL + centerPoint;

		vertices->push_back(output[0]);
		vertices->push_back(output[2]);
		vertices->push_back(output[3]);
		vertices->push_back(output[0]);
		vertices->push_back(output[1]);
		vertices->push_back(output[2]);
    }

    void ShapeCreator::createTriangle(std::vector<glm::vec3>* vertices, const glm::vec3& up, const glm::vec3& right, const glm::vec3& left, glm::vec3 centerPoint)
    {
		const glm::vec3 horizontal = right - left;
		const glm::vec3 vertical = up - left;
		//const glm::vec3 normal = cross(horizontal, vertical);
		glm::vec3 output[3];

		/*for (int i = 0; i < 3; i++) {
			output[i].Normal = normal;
		}*/

		output[0] = up + centerPoint;
		output[1] = right + centerPoint;
		output[2] = left + centerPoint;

		vertices->push_back(output[0]);
		vertices->push_back(output[2]);
		vertices->push_back(output[1]);
    }
}
