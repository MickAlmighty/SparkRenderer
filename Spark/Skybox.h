#pragma once

#include "Component.h"
#include "Structs.h"
#include "Texture.h"

namespace spark
{
class Skybox : public Component
{
    public:
    Skybox();
    ~Skybox() override;

    void update() override;
    void drawUIBody() override;

    std::string getSkyboxName() const;
    void loadSkyboxByName(std::string name);

    void setActiveSkybox(bool isActive);
    bool isSkyboxActive() const;

    static Skybox* getActiveSkybox();

    private:
    void onActive() override;
    void onInactive() override;

    void createPbrCubemap(const std::shared_ptr<resources::Texture>& texture);

    void setSkyboxInScene(const std::shared_ptr<PbrCubemapTexture>& skyboxCubemap) const;

    inline static Skybox* activeSkyboxPtr{nullptr};

    std::string skyboxName{};
    std::shared_ptr<PbrCubemapTexture> pbrCubemapTexture{nullptr};
    RTTR_ENABLE(Component)
};
}  // namespace spark
