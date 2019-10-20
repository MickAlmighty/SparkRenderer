#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <glm/glm.hpp>

#include "Component.h"

namespace spark {

	struct DirectionalLightData;

	class DirectionalLight : public Component
	{
	public:
		DirectionalLightData getLightData();

		bool getDirty() const;
		glm::vec3 getDirection() const;
		glm::vec3 getColor() const;
		void setDirection(glm::vec3 direction_);
		void setColor(glm::vec3 color_);

		DirectionalLight(std::string name_ = "DirectionalLight");
		virtual ~DirectionalLight() = default;

		void setActive(bool active_) override;
		SerializableType getSerializableType() override;
		Json::Value serialize() override;
		void deserialize(Json::Value& root) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool dirty = true;
		bool addedToLigtManager = false;
		glm::vec3 direction{0};
		glm::vec3 color{0};
	};
}

#endif