#include "SpotLight.h"

#include <glm/gtc/type_ptr.hpp>

#include "GameObject.h"
#include "JsonSerializer.h"
#include "Structs.h"

namespace spark {

	SpotLightData SpotLight::getLightData() const
	{	
		SpotLightData data{};
		data.direction = getDirection();
		data.position = getPosition();
		data.color = getColor() * getColorStrength();
		data.cutOff = glm::cos(glm::radians(getCutOff()));
		data.outerCutOff = glm::cos(glm::radians(getOuterCutOff()));
		return data;
	}

	bool SpotLight::getDirty() const
	{
		return dirty;
	}

	glm::vec3 SpotLight::getPosition() const
	{
		return getGameObject()->transform.world.getPosition();
	}

	glm::vec3 SpotLight::getDirection() const
	{
		return direction;
	}

	glm::vec3 SpotLight::getColor() const
	{
		return color;
	}

	float SpotLight::getColorStrength() const
	{
		return colorStrength;
	}

	float SpotLight::getCutOff() const
	{
		return cutOff;
	}

	float SpotLight::getOuterCutOff() const
	{
		return outerCutOff;
	}

	void SpotLight::resetDirty()
	{
		dirty = false;
	}

	void SpotLight::setColor(glm::vec3 color_)
	{
		dirty = true;
		color = color_;
	}

	void SpotLight::setColorStrength(float strength)
	{
		dirty = true;
		colorStrength = strength;
	}

	void SpotLight::setDirection(glm::vec3 direction_)
	{
		dirty = true;
		direction = direction_;
	}

	void SpotLight::setCutOff(float cutOff_)
	{
		if (cutOff_ < 0.0f) return;
		if (cutOff_ > 360.0f) return;
		
		dirty = true;
		cutOff = cutOff_;
	}

	void SpotLight::setOuterCutOff(float outerCutOff_)
	{
		dirty = true;
		outerCutOff = outerCutOff_;
	}

	SpotLight::SpotLight(std::string name_) : Component(name_)
	{
	}

	void SpotLight::setActive(bool active_)
	{
		dirty = true;
		active = active_;
	}

	SerializableType SpotLight::getSerializableType()
	{
		return SerializableType::SSpotLight;
	}

	Json::Value SpotLight::serialize()
	{
		Json::Value root;
		root["color"] = JsonSerializer::serializeVec3(color);
		root["colorStrength"] = colorStrength;
		root["cutOff"] = cutOff;
		root["outerCutOff"] = outerCutOff;
		root["direction"] = JsonSerializer::serializeVec3(direction);
		return root;
	}

	void SpotLight::deserialize(Json::Value& root)
	{
		color = JsonSerializer::deserializeVec3(root["color"]);
		colorStrength = root.get("colorStrength", 1.0f).asFloat();
		cutOff = root.get("cutOff", 30.0f).asFloat();
		outerCutOff = root.get("outerCutOff", 45.0f).asFloat();
		direction = JsonSerializer::deserializeVec3(root["direction"]);
	}

	void SpotLight::update()
	{
		if (!addedToLightManager)
		{
			SceneManager::getInstance()->getCurrentScene()->lightManager->addSpotLight(shared_from_base<SpotLight>());
			addedToLightManager = true;
		}

		const glm::vec3 newPos = getPosition();
		if (newPos != lastPos)
		{
			dirty = true;
		}
		lastPos = newPos;
	}

	void SpotLight::fixedUpdate()
	{

	}

	void SpotLight::drawGUI()
	{
		glm::vec3 colorToEdit = getColor();
		float colorStrengthToEdit = getColorStrength();
		float cutOffToEdit = getCutOff();
		float outerCutOffToEdit = getOuterCutOff();
		glm::vec3 directionToEdit = getDirection();
		ImGui::ColorEdit3("color", glm::value_ptr(colorToEdit));
		ImGui::DragFloat("colorStrength", &colorStrengthToEdit, 0.01f);
		ImGui::DragFloat("cutOff", &cutOffToEdit, 1.0f, 0.0f, 180.0f);
		ImGui::DragFloat("outerCutOff", &outerCutOffToEdit, 1.0f, 0.0f, 180.0f);
		ImGui::SliderFloat3("direction", glm::value_ptr(directionToEdit), -1.0f, 1.0f);

		if (colorStrengthToEdit < 0)
		{
			colorStrengthToEdit = 0;
		}

		if (colorToEdit != getColor())
		{
			setColor(colorToEdit);
		}

		if (colorStrengthToEdit != getColorStrength())
		{
			setColorStrength(colorStrengthToEdit);
		}

		if (cutOffToEdit != getCutOff())
		{
			setCutOff(cutOffToEdit);
		}
		
		if (outerCutOffToEdit != getOuterCutOff())
		{
			setOuterCutOff(outerCutOffToEdit);
		}

		if (directionToEdit != getDirection())
		{
			setDirection(directionToEdit);
		}

		removeComponentGUI<SpotLight>();
	}
}


