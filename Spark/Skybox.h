#pragma once
#include "Component.h"
#include "Texture.h"

namespace spark
{
class Skybox : public Component
{
    public:
    Skybox();
    ~Skybox() override;
    void update() override;
    void fixedUpdate() override;
    void createPbrCubemap(const std::shared_ptr<resources::Texture>& texture);
    void drawGUI() override;

    void setActiveSkybox(bool isActive);
    bool isSkyboxActive() const;

    private:
    void setActive(bool active_) override;
    std::string getSkyboxName() const;
    void loadSkyboxName(std::string name);

    inline static std::vector<Skybox*> skyBoxes{};

    bool activeSkybox{false};
    std::string skyboxName{};
    std::shared_ptr<PbrCubemapTexture> pbrCubemapTexture{nullptr};
    RTTR_REGISTRATION_FRIEND;
    RTTR_ENABLE(Component)
};
}  // namespace spark
