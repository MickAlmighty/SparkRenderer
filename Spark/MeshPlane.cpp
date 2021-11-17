#include "MeshPlane.h"

#include "GameObject.h"
#include "GUI/SparkGui.h"
#include "ReflectionUtils.h"
#include "renderers/RenderingRequest.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
void MeshPlane::setup()
{
    const std::vector<glm::vec3> vertices{{-1.0f, 1.0f, 0.0f}, {1.0f, 1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {-1.0f, -1.0f, 0.0f}};

    const std::vector<glm::vec2> texCoords{{0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}};

    const auto positionAttribute = VertexAttribute::createVertexShaderAttributeInfo(0, 3, vertices);
    const auto texCoordsAttribute = VertexAttribute::createVertexShaderAttributeInfo(1, 2, texCoords);
    auto vertexAttributes = std::vector<VertexAttribute>{positionAttribute, texCoordsAttribute};
    std::vector<unsigned int> indices{0, 1, 2, 2, 3, 0};
    auto textures = std::map<TextureTarget, std::shared_ptr<resources::Texture>>{};
    planeMesh = std::make_shared<Mesh>(vertexAttributes, indices, textures, "Mesh", ShaderType::COLOR_ONLY);
}

MeshPlane::MeshPlane() : Component()
{
    setup();
}

void MeshPlane::update()
{
    renderers::RenderingRequest request{};
    request.shaderType = ShaderType::COLOR_ONLY;
    request.gameObject = getGameObject();
    request.mesh = planeMesh;
    request.model = getGameObject()->transform.world.getMatrix();

    getGameObject()->getScene()->addRenderingRequest(request);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::MeshPlane>("MeshPlane")
        .constructor()(rttr::policy::ctor::as_std_shared_ptr)
        .property("planeMesh", &spark::MeshPlane::planeMesh)(rttr::detail::metadata(spark::SerializerMeta::Serializable, false));
}