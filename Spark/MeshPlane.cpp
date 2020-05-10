#include "MeshPlane.h"

#include "EngineSystems/ResourceManager.h"
#include "EngineSystems/SparkRenderer.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "ReflectionUtils.h"
#include "RenderingRequest.h"
#include "Shader.h"

namespace spark
{
void MeshPlane::setup()
{
    const std::vector<glm::vec3> vertices{
        {-1.0f, 1.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {1.0f, -1.0f, 0.0f},
        {-1.0f, -1.0f, 0.0f}
    };

    const std::vector<glm::vec2> texCoords{
        {0.0f, 1.0f},
        {1.0f, 1.0f},
        {1.0f, 0.0f},
        {0.0f, 0.0f}
    };

    std::vector<unsigned int> indices{0, 1, 2, 2, 3, 0};

    const auto positionAttribute = VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, vertices);
    const auto texCoordsAttribute = VertexShaderAttribute::createVertexShaderAttributeInfo(1, 2, texCoords);
    planeMesh = std::make_shared<Mesh>(std::vector<VertexShaderAttribute>{positionAttribute, texCoordsAttribute}, indices,
                                    std::map<TextureTarget, std::shared_ptr<resources::Texture>>{}, "Mesh", ShaderType::SOLID_COLOR_SHADER);
    planeMesh->gpuLoad();
}

MeshPlane::MeshPlane() : Component("MeshPlane")
{
    setup();
}

MeshPlane::MeshPlane(std::string&& name) : Component(std::move(name))
{
    setup();
}

MeshPlane::~MeshPlane()
{
    planeMesh->gpuUnload();
}


void MeshPlane::update()
{
    RenderingRequest request{};
    request.shaderType = ShaderType::DEFAULT_SHADER;
    request.gameObject = getGameObject();
    request.mesh = planeMesh;
    request.model = getGameObject()->transform.world.getMatrix();

    SparkRenderer::getInstance()->addRenderingRequest(request);
}

void MeshPlane::fixedUpdate() {}

void MeshPlane::drawGUI()
{
    removeComponentGUI<MeshPlane>();
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::MeshPlane>("MeshPlane")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("planeMesh", &spark::MeshPlane::planeMesh)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false));
}