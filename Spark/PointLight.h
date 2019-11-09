#pragma once

#include <glm/glm.hpp>

#include "Component.h"

namespace spark {
	
	struct PointLightData;

	class PointLight final : public Component
	{
	public:
		PointLight();
		~PointLight() = default;
        PointLight(const PointLight&) = delete;
        PointLight(const PointLight&&) = delete;
        PointLight& operator=(const PointLight&) = delete;
        PointLight& operator=(const PointLight&&) = delete;

	    PointLightData getLightData() const;
		bool getDirty() const;
		glm::vec3 getPosition() const;
		glm::vec3 getColor() const;
		float getColorStrength() const;
		void resetDirty();
		void setColor(glm::vec3 color_);
		void setColorStrength(float strength);
		void setActive(bool active_) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool dirty = true;
		bool addedToLightManager = false;
		glm::vec3 color{ 1 };
		float colorStrength{ 1 };
		glm::vec3 lastPos{0};
        RTTR_REGISTRATION_FRIEND;
        RTTR_ENABLE(Component);
	};
}

