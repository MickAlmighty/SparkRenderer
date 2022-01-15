#include "ClusterBasedLightCullingPass.hpp"

#include "CommonUtils.h"
#include "EditorCamera.hpp"
#include "Spark.h"

namespace spark::renderers
{
ClusterBasedLightCullingPass::ClusterBasedLightCullingPass(unsigned int width, unsigned int height) : w(width), h(height)
{
    clusterCreationShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("clusterCreation.glsl");
    determineActiveClustersShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("determineActiveCluster.glsl");
    buildCompactClusterListShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("buildCompactClusterList.glsl");
    clusterBasedLightCullingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("clusterBasedLightCulling.glsl");

    clusters.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(glm::vec4) * 2);
    activeClusters.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint));
    activeClusterIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint));
    perClusterGlobalLightIndicesBufferMetadata.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 6);
    globalPointLightIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);
    globalSpotLightIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);
    globalLightProbeIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);

    const std::array<unsigned int, 3> dispatches{1, 1, 1};
    activeClustersCount.updateData(dispatches);

    resize(w, h);
}

void ClusterBasedLightCullingPass::process(GLuint depthTexture, const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    PUSH_DEBUG_GROUP(CLUSTER_BASED_LIGHT_CULLING)

    createClusters(scene, camera);
    determineActiveClusters(depthTexture, scene, camera);
    buildCompactClusterList();
    lightCulling(scene, camera);

    clearActiveClustersCounter();
    POP_DEBUG_GROUP()
}

void ClusterBasedLightCullingPass::createClusters(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    const bool hasNearAndFarPlaneChanged = lastCamNearZ != camera->zNear || lastCamFarZ != camera->zFar;
    const bool hasTileDimensionChanged = lastPxTileSize != pxTileSize;
    //if(hasNearAndFarPlaneChanged || hasTileDimensionChanged)
    {
        clusterCreationShader->use();
        clusterCreationShader->setVec2("tileSize", pxTileSize);
        clusterCreationShader->bindSSBO("ClusterData", clusters);
        clusterCreationShader->bindUniformBuffer("Camera", camera->getUbo());
        clusterCreationShader->dispatchCompute(utils::uiCeil(dispatchSize.x, 32u), utils::uiCeil(dispatchSize.y, 32u), dispatchSize.z);

        lastCamNearZ = camera->zNear;
        lastCamFarZ = camera->zFar;
        lastPxTileSize = pxTileSize;
    }
}

void ClusterBasedLightCullingPass::determineActiveClusters(GLuint depthTexture, const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    determineActiveClustersShader->use();
    determineActiveClustersShader->setVec2("tileSize", pxTileSize);
    determineActiveClustersShader->bindUniformBuffer("Camera", camera->getUbo());
    determineActiveClustersShader->bindSSBO("ActiveClusters", activeClusters);
    determineActiveClustersShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", perClusterGlobalLightIndicesBufferMetadata);

    glBindTextureUnit(0, depthTexture);
    determineActiveClustersShader->dispatchCompute(utils::uiCeil(w, 32u), utils::uiCeil(h, 32u), 1);
    glBindTextures(0, 1, nullptr);
}

void ClusterBasedLightCullingPass::buildCompactClusterList()
{
    buildCompactClusterListShader->use();
    buildCompactClusterListShader->bindSSBO("ActiveClusters", activeClusters);
    buildCompactClusterListShader->bindSSBO("ActiveClustersCount", activeClustersCount);
    buildCompactClusterListShader->bindSSBO("ActiveClusterIndices", activeClusterIndices);
    buildCompactClusterListShader->dispatchCompute(utils::uiCeil(dispatchSize.x, 16u), utils::uiCeil(dispatchSize.y, 16u), dispatchSize.z);
}

void ClusterBasedLightCullingPass::lightCulling(const std::shared_ptr<Scene>& scene, const std::shared_ptr<ICamera>& camera)
{
    clusterBasedLightCullingShader->use();
    clusterBasedLightCullingShader->bindUniformBuffer("Camera", camera->getUbo());
    clusterBasedLightCullingShader->bindSSBO("ActiveClusterIndices", activeClusterIndices);
    clusterBasedLightCullingShader->bindSSBO("ClusterData", clusters);
    clusterBasedLightCullingShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", perClusterGlobalLightIndicesBufferMetadata);
    clusterBasedLightCullingShader->bindSSBO("GlobalPointLightIndices", globalPointLightIndices);
    clusterBasedLightCullingShader->bindSSBO("GlobalSpotLightIndices", globalSpotLightIndices);
    clusterBasedLightCullingShader->bindSSBO("GlobalLightProbeIndices", globalLightProbeIndices);
    clusterBasedLightCullingShader->bindSSBO("PointLightData", scene->lightManager->getPointLightSSBO());
    clusterBasedLightCullingShader->bindSSBO("SpotLightData", scene->lightManager->getSpotLightSSBO());
    clusterBasedLightCullingShader->bindSSBO("LightProbeData", scene->lightManager->getLightProbeSSBO());
    clusterBasedLightCullingShader->dispatchComputeIndirect(activeClustersCount.ID);
}

void ClusterBasedLightCullingPass::clearActiveClustersCounter()
{
    const std::array<GLuint, 1> globalActiveClusterCount{0};
    activeClustersCount.updateSubData(0, globalActiveClusterCount);
}

void ClusterBasedLightCullingPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    pxTileSize = glm::vec2(w, h) / glm::vec2(dispatchSize);
}
}  // namespace spark::renderers
