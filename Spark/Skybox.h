#pragma once

#include "Component.h"
#include "PbrCubemapTexture.hpp"
#include "Texture.h"

namespace spark
{
class Skybox : public Component
{
    public:
    ~Skybox() override;

    void update() override;
    void drawUIBody() override;

    std::string getSkyboxPath() const;
    void loadSkybox(std::string relativePath);

    void setActiveSkybox(bool isActive);
    bool isSkyboxActive() const;

    static Skybox* getActiveSkybox();

    private:
    Skybox() = default;

    void onActive() override;
    void onInactive() override;

    void createPbrCubemap(const std::shared_ptr<resources::Texture>& texture);

    void setSkyboxInScene(const std::shared_ptr<PbrCubemapTexture>& skyboxCubemap) const;

    inline static Skybox* activeSkyboxPtr{nullptr};

    std::string skyboxPath{};
    std::shared_ptr<PbrCubemapTexture> pbrCubemapTexture{nullptr};

    RTTR_REGISTRATION_FRIEND
    RTTR_ENABLE(Component)
};
}  // namespace spark
