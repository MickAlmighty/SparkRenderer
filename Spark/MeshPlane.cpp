#include "MeshPlane.h"

#include "EngineSystems/SparkRenderer.h"
#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "ReflectionUtils.h"
#include "RenderingRequest.h"
#include "Shader.h"
#include "Spark.h"

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


    const auto positionAttribute = VertexShaderAttribute::createVertexShaderAttributeInfo(0, 3, vertices);
    const auto texCoordsAttribute = VertexShaderAttribute::createVertexShaderAttributeInfo(1, 2, texCoords);
    auto vertexShaderAttributes = std::vector<VertexShaderAttribute>{positionAttribute, texCoordsAttribute};
    std::vector<unsigned int> indices{0, 1, 2, 2, 3, 0};
    auto textures = std::map<TextureTarget, std::shared_ptr<resources::Texture>>{};
    planeMesh = std::make_shared<Mesh>(vertexShaderAttributes, indices,
                                    textures, "Mesh", ShaderType::COLOR_ONLY);
}

MeshPlane::MeshPlane() : Component("MeshPlane")
{
    setup();
}

MeshPlane::MeshPlane(std::string&& name) : Component(std::move(name))
{
    setup();
}

void MeshPlane::update()
{
    RenderingRequest request{};
    request.shaderType = ShaderType::COLOR_ONLY;
    request.gameObject = getGameObject();
    request.mesh = planeMesh;
    request.model = getGameObject()->transform.world.getMatrix();

    Spark::get().getRenderer().addRenderingRequest(request);
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