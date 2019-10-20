#include "DirectionalLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "EngineSystems/SceneManager.h"
#include "GameObject.h"
#include "JsonSerializer.h"
#include "Structs.h"

namespace spark {


	DirectionalLightData DirectionalLight::getLightData()
	{
		dirty = false;
		return { direction, color * colorStrength };
	}

	bool DirectionalLight::getDirty() const
	{
		return dirty;
	}

	glm::vec3 DirectionalLight::getDirection() const
	{
		return direction;
	}

	glm::vec3 DirectionalLight::getColor() const
	{
		return color;
	}

	float DirectionalLight::getColorStrength() const
	{
		return colorStrength;
	}

	void DirectionalLight::resetDirty()
	{
		dirty = false;
	}

	void DirectionalLight::setDirection(glm::vec3 direction_)
	{
		dirty = true;
		direction = direction_;
	}

	void DirectionalLight::setColor(glm::vec3 color_)
	{
		dirty = true;
		color = color_;
	}

	void DirectionalLight::setColorStrength(float strength)
	{
		dirty = true;
		colorStrength = strength;
	}

	DirectionalLight::DirectionalLight(std::string name_) : Component(name_)
	{
		
	}

	void DirectionalLight::setActive(bool active_)
	{
		dirty = true;
		active = active_;
	}

	SerializableType DirectionalLight::getSerializableType()
	{
		return SerializableType::SDirectionalLight;
	}

	Json::Value DirectionalLight::serialize()
	{
		Json::Value root;
		root["name"] = name;
		root["direction"] = JsonSerializer::serializeVec3(direction);
		root["color"] = JsonSerializer::serializeVec3(color);
		root["colorStrength"] = colorStrength;
		return root;
	}

	void DirectionalLight::deserialize(Json::Value& root)
	{
		name = root.get("name", "DirectionalLight").asString();
		direction = JsonSerializer::deserializeVec3(root["direction"]);
		color = JsonSerializer::deserializeVec3(root["color"]);
		colorStrength = root.get("colorStrength", 1.0f).asFloat();
	}

	void DirectionalLight::update()
	{
		if(!addedToLightManager)
		{
			SceneManager::getInstance()->getCurrentScene()->lightManager->addDirectionalLight(shared_from_base<DirectionalLight>());
			addedToLightManager = true;
		}
		
	}

	void DirectionalLight::fixedUpdate()
	{

	}

	void DirectionalLight::drawGUI()
	{
		glm::vec3 colorToEdit = getColor();
		glm::vec3 directionToEdit = getDirection();
		float colorStrengthToEdit = getColorStrength();
		ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
		ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
		ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);

		if (colorStrengthToEdit < 0)
		{
			colorStrengthToEdit = 0;
		}

		if (directionToEdit != getDirection())
		{
			setDirection(directionToEdit);
		}
		if (colorToEdit != getColor())
		{
			setColor(colorToEdit);
		}
		if (colorStrengthToEdit != getColorStrength())
		{
			setColorStrength(colorStrengthToEdit);
		}
		removeComponentGUI<DirectionalLight>();
	}

}
