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
		return { direction, color };
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
		return root;
	}

	void DirectionalLight::deserialize(Json::Value& root)
	{
		name = root.get("name", "DirectionalLight").asString();
		direction = JsonSerializer::deserializeVec3(root["direction"]);
		color = JsonSerializer::deserializeVec3(root["color"]);
	}

	void DirectionalLight::update()
	{
		if(!addedToLigtManager)
		{
			SceneManager::getInstance()->getCurrentScene()->lightManager->addDirectionalLight(shared_from_base<DirectionalLight>());
			addedToLigtManager = true;
		}
		
	}

	void DirectionalLight::fixedUpdate()
	{

	}

	void DirectionalLight::drawGUI()
	{
		glm::vec3 colorToEdit = getColor();
		glm::vec3 directionToEdit = getDirection();
		ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
		ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);

		if (directionToEdit != getDirection())
		{
			setDirection(directionToEdit);
		}
		if (colorToEdit != getColor())
		{
			setColor(colorToEdit);
		}
		removeComponentGUI<DirectionalLight>();
	}

}
