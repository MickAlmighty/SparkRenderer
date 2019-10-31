#ifndef SPOT_LIGHT_H
#define SPOT_LIGHT_H

#include <glm/vec3.hpp>

#include "Component.h"
#include "ISerializable.h"

namespace spark {
	struct SpotLightData;

	class SpotLight : public Component
	{
	public:
		SpotLightData getLightData() const;

		bool getDirty() const;
		void resetDirty();
		glm::vec3 getPosition() const;
		glm::vec3 getDirection() const;
		glm::vec3 getColor() const;
		float getColorStrength() const;
		float getCutOff() const;
		float getOuterCutOff() const;

		void setColor(glm::vec3 color_);
		void setColorStrength(float strength);
		void setDirection(glm::vec3 direction_);
		void setCutOff(float cutOff_);
		void setOuterCutOff(float outerCutOff_);

		SpotLight(std::string name_ = "SpotLight");
		virtual ~SpotLight() = default;

		void setActive(bool active_) override;
		void update() override;
		void fixedUpdate() override;
		void drawGUI() override;

	private:
		bool dirty = true;
		bool addedToLightManager = false;

		glm::vec3 color{ 1 };
		float colorStrength{ 1 };
		glm::vec3 direction{ 0.0f, -1.0f, 0.0f };
		float cutOff{ 30.0f };
		float outerCutOff{ 45.0f };

		glm::vec3 lastPos{ 0 };
	};
}

#endif