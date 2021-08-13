#include "ClusterBasedLightCullingPass.hpp"

#include "CommonUtils.h"
#include "Logging.h"
#include "Spark.h"

namespace spark
{
ClusterBasedLightCullingPass::ClusterBasedLightCullingPass(unsigned int width, unsigned int height, const UniformBuffer& cameraUbo,
                                                           const std::shared_ptr<lights::LightManager>& lightManager)
    : w(width), h(height)
{
    clusterCreationShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("clusterCreation.glsl");
    clusterCreationShader->bindSSBO("ClusterData", clusters);
    clusterCreationShader->bindUniformBuffer("Camera", cameraUbo);

    determineActiveClustersShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("determineActiveCluster.glsl");
    determineActiveClustersShader->bindUniformBuffer("Camera", cameraUbo);
    determineActiveClustersShader->bindSSBO("ActiveClusters", activeClusters);
    determineActiveClustersShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", perClusterGlobalLightIndicesBufferMetadata);

    buildCompactClusterListShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("buildCompactClusterList.glsl");
    buildCompactClusterListShader->bindSSBO("ActiveClusters", activeClusters);
    buildCompactClusterListShader->bindSSBO("ActiveClustersCount", activeClustersCount);
    buildCompactClusterListShader->bindSSBO("ActiveClusterIndices", activeClusterIndices);

    clusterBasedLightCullingShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("clusterBasedLightCulling.glsl");
    clusterBasedLightCullingShader->bindUniformBuffer("Camera", cameraUbo);
    clusterBasedLightCullingShader->bindSSBO("ActiveClusterIndices", activeClusterIndices);
    clusterBasedLightCullingShader->bindSSBO("ClusterData", clusters);
    clusterBasedLightCullingShader->bindSSBO("PerClusterGlobalLightIndicesBufferMetadata", perClusterGlobalLightIndicesBufferMetadata);
    clusterBasedLightCullingShader->bindSSBO("GlobalLightIndicesOffset", globalLightIndicesOffset);
    clusterBasedLightCullingShader->bindSSBO("GlobalPointLightIndices", globalPointLightIndices);
    clusterBasedLightCullingShader->bindSSBO("GlobalSpotLightIndices", globalSpotLightIndices);
    clusterBasedLightCullingShader->bindSSBO("GlobalLightProbeIndices", globalLightProbeIndices);

    clusters.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(glm::vec4) * 2);
    activeClusters.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint));
    activeClusterIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint));
    perClusterGlobalLightIndicesBufferMetadata.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 6);
    globalLightIndicesOffset.resizeBuffer(sizeof(GLuint) * 3);
    globalPointLightIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);
    globalSpotLightIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);
    globalLightProbeIndices.resizeBuffer(dispatchSize.x * dispatchSize.y * dispatchSize.z * sizeof(GLuint) * 256);

    const std::array<unsigned int, 3> dispatches{1, 1, 1};
    activeClustersCount.updateData(dispatches);

    bindLightBuffers(lightManager);
    resize(w, h);
}

void ClusterBasedLightCullingPass::process(GLuint depthTexture)
{
    PUSH_DEBUG_GROUP(CLUSTER_BASED_LIGHT_CULLING)
    //globalPointLightIndices.clearData();
    //globalSpotLightIndices.clearData();
    //globalLightProbeIndices.clearData();

    clusterCreationShader->use();
    clusterCreationShader->setVec2("tileSize", pxTileSize);
    clusterCreationShader->dispatchCompute(utils::uiCeil(dispatchSize.x, 32u), utils::uiCeil(dispatchSize.y, 32u), dispatchSize.z);

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    {
        /*const float logNearByFar = glm::log(camera->getFarPlane() / camera->getNearPlane());
        const float equation3Part1 = static_cast<float>(dispatchSize.z) / logNearByFar;
        const float equation3Part2 = static_cast<float>(dispatchSize.z) * glm::log(camera->getNearPlane()) / logNearByFar;*/

        determineActiveClustersShader->use();
        determineActiveClustersShader->setVec2("tileSize", pxTileSize);
       /* determineActiveClustersShader->setFloat("equation3Part1", equation3Part1);
        determineActiveClustersShader->setFloat("equation3Part2", equation3Part2);*/

        glBindTextureUnit(0, depthTexture);
        determineActiveClustersShader->dispatchCompute(utils::uiCeil(w, 32u), utils::uiCeil(h, 32u), 1);
        glBindTextures(0, 1, nullptr);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    {
        buildCompactClusterListShader->use();
        buildCompactClusterListShader->dispatchCompute(utils::uiCeil(dispatchSize.x, 32u), utils::uiCeil(dispatchSize.y, 32u), dispatchSize.z);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    {
        clusterBasedLightCullingShader->use();
        clusterBasedLightCullingShader->dispatchComputeIndirect(activeClustersCount.ID);
    }

    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

    const std::array<GLuint, 1> globalActiveClusterCount{0};
    activeClustersCount.updateSubData(0, globalActiveClusterCount);
    // activeClustersCount.clearData();
    globalLightIndicesOffset.clearData();
    activeClusterIndices.clearData(); // if culling will work, to be removed
    POP_DEBUG_GROUP()
}

void ClusterBasedLightCullingPass::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    pxTileSize = glm::vec2(w, h) / glm::vec2(dispatchSize);
}

void ClusterBasedLightCullingPass::bindLightBuffers(const std::shared_ptr<lights::LightManager>& lightManager)
{
    clusterBasedLightCullingShader->bindSSBO("PointLightData", lightManager->getPointLightSSBO());
    clusterBasedLightCullingShader->bindSSBO("SpotLightData", lightManager->getSpotLightSSBO());
    clusterBasedLightCullingShader->bindSSBO("LightProbeData", lightManager->getLightProbeSSBO());
}
}  // namespace spark
