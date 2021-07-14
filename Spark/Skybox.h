#pragma once
#include "Component.h"
#include "Texture.h"

namespace spark
{
class Skybox : public Component
{
    public:
    Skybox();
    virtual ~Skybox() override;
    void update() override;
    void fixedUpdate() override;
    void drawGUI() override;

    std::string getSkyboxName() const;
    void loadSkyboxByName(std::string name);

    void setActiveSkybox(bool isActive);
    bool isSkyboxActive() const;

    static Skybox* getActiveSkybox();

    private:
    void onActive() override;
    void onInactive() override;

    void createPbrCubemap(const std::shared_ptr<resources::Texture>& texture);

    inline static Skybox* activeSkyboxPtr{nullptr};

    bool activeSkybox{false};
    std::string skyboxName{};
    std::shared_ptr<PbrCubemapTexture> pbrCubemapTexture{nullptr};
    RTTR_ENABLE(Component)
};
}  // namespace spark
