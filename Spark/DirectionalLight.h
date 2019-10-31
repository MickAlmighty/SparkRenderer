#ifndef DIRECTIONAL_LIGHT_H
#define DIRECTIONAL_LIGHT_H

#include <glm/glm.hpp>

#include "Component.h"

namespace spark {

	struct DirectionalLightData;

	class DirectionalLight : public Component
	{
	public:
		DirectionalLightData getLightData() const;

		bool getDirty() const;
		glm::vec3 getDirection() const;
		glm::vec3 getColor() const;
		float getColorStrength() const;
		void resetDirty();
		void setDirection(glm::vec3 direction_);
		void setColor(glm::vec3 color_);
		void setColorStrength(float strength);

		DirectionalLight(std::string name_ = "DirectionalLight");
		virtual ~DirectionalLight() = default;

		void setActive(bool active_) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool dirty = true;
		bool addedToLightManager = false;
		glm::vec3 direction{0.0f, -1.0f, 0.0f};
		glm::vec3 color{1};
		float colorStrength{ 1 };
	};
}

#endif