#pragma once

#include <glm/glm.hpp>

#include "Component.h"

namespace spark {
	
	struct PointLightData;

	class PointLight : public Component
	{
	public:
		PointLightData getLightData() const;

		bool getDirty() const;
		glm::vec3 getPosition() const;
		glm::vec3 getColor() const;
		float getColorStrength() const;
		void resetDirty();
		void setColor(glm::vec3 color_);
		void setColorStrength(float strength);

		PointLight(std::string name_ = "PointLight");
		virtual ~PointLight() = default;

		void setActive(bool active_) override;
		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool dirty = true;
		bool addedToLightManager = false;
		
		glm::vec3 color{ 0 };
		float colorStrength{ 1 };

		glm::vec3 lastPos{0};
	};
}

