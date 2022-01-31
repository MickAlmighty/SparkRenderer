#include "PbrCubemapTexture.hpp"

#include "utils/CommonUtils.h"
#include "Cube.hpp"
#include "ScreenQuad.hpp"
#include "Shader.h"
#include "Spark.h"
#include "Timer.h"

namespace spark
{
PbrCubemapTexture::PbrCubemapTexture(GLuint hdrTexture, unsigned size)
{
    setup(hdrTexture, size);
}

void PbrCubemapTexture::setup(GLuint hdrTexture, unsigned cubemapSize)
{
    Timer t("HDR PBR cubemap generation time");

    SSBO cubemapViewMatrices{};
    cubemapViewMatrices.resizeBuffer(sizeof(glm::mat4) * 6);
    cubemapViewMatrices.updateData(utils::getCubemapViewMatrices(glm::vec3(0)));

    Cube cube = Cube();

    GLuint captureFBO;
    glGenFramebuffers(1, &captureFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);

    // these shaders are created in SparkRenderer with uniforms and buffers already bound
    const std::shared_ptr<resources::Shader> resampleCubemapShader =
        Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("resampleCubemap.glsl");
    const auto equirectangularToCubemapShader =
        Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("equirectangularToCubemap.glsl");
    const auto irradianceShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("irradiance.glsl");
    const auto prefilterShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("prefilter.glsl");

    irradianceShader->bindSSBO("Views", cubemapViewMatrices);
    prefilterShader->bindSSBO("Views", cubemapViewMatrices);
    resampleCubemapShader->bindSSBO("Views", cubemapViewMatrices);
    equirectangularToCubemapShader->bindSSBO("Views", cubemapViewMatrices);

    const auto envCubemap = createEnvironmentCubemapWithMipmapChain(captureFBO, hdrTexture, cubemapSize, cube, equirectangularToCubemapShader);
    this->irradianceCubemap = createIrradianceCubemap(captureFBO, envCubemap, cube, irradianceShader);
    this->prefilteredCubemap = createPreFilteredCubemap(captureFBO, envCubemap, cubemapSize, cube, prefilterShader, resampleCubemapShader);

    // creating cubemap in the line below is meant to reduce memory usage by getting rid of mipmaps
    this->cubemap = createCubemapAndCopyDataFromFirstLayerOf(envCubemap, cubemapSize);
    // this->brdfLUTTexture = createBrdfLookupTexture(captureFBO, 1024);

    glDeleteFramebuffers(1, &captureFBO);
}

utils::TextureHandle PbrCubemapTexture::createEnvironmentCubemapWithMipmapChain(
    GLuint framebuffer, GLuint equirectangularTexture, unsigned size, Cube& cube,
    const std::shared_ptr<resources::Shader>& equirectangularToCubemapShader)
{
    PUSH_DEBUG_GROUP(EQUIRECTANGULAR_TO_CUBEMAP_WITH_MIPS);

    utils::TextureHandle envCubemap = utils::createCubemap(size, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    glViewport(0, 0, size, size);

    equirectangularToCubemapShader->use();
    glBindTextureUnit(0, equirectangularTexture);

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, envCubemap.get(), 0);
    cube.draw();

    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap.get());
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    POP_DEBUG_GROUP();
    return envCubemap;
}

utils::TextureHandle PbrCubemapTexture::createIrradianceCubemap(GLuint framebuffer, utils::TextureHandle environmentCubemap, Cube& cube,
                                                                const std::shared_ptr<resources::Shader>& irradianceShader)
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP);

    utils::TextureHandle irradianceMap = utils::createCubemap(32, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, 32, 32);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap.get());

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, irradianceMap.get(), 0);
    cube.draw();

    POP_DEBUG_GROUP();

    return irradianceMap;
}

utils::TextureHandle PbrCubemapTexture::createPreFilteredCubemap(GLuint framebuffer, utils::TextureHandle environmentCubemap, unsigned envCubemapSize,
                                                                 Cube& cube, const std::shared_ptr<resources::Shader>& prefilterShader,
                                                                 const std::shared_ptr<resources::Shader>& resampleCubemapShader)
{
    PUSH_DEBUG_GROUP(PREFILTER_CUBEMAP);

    constexpr unsigned int prefilteredMapSize = 128;
    utils::TextureHandle prefilteredMap =
        utils::createCubemap(prefilteredMapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    {
        resampleCubemapShader->use();
        glBindTextureUnit(0, environmentCubemap.get());

        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilteredMap.get(), 0);

        glViewport(0, 0, prefilteredMapSize, prefilteredMapSize);
        cube.draw();
    }

    const GLuint maxMipLevels = 5;
    prefilterShader->use();
    glBindTextureUnit(0, environmentCubemap.get());
    prefilterShader->setFloat("textureSize", static_cast<float>(envCubemapSize));

    for(unsigned int mip = 1; mip < maxMipLevels; ++mip)
    {
        const auto mipSize = static_cast<unsigned int>(prefilteredMapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipSize, mipSize);
        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilteredMap.get(), mip);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);

        cube.draw();
    }

    POP_DEBUG_GROUP();

    return prefilteredMap;
}

utils::TextureHandle PbrCubemapTexture::createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize,
                                                                const std::shared_ptr<resources::Shader>& brdfShader)
{
    PUSH_DEBUG_GROUP(BRDF_LOOKUP_TABLE);

    utils::TextureHandle brdfLUTTexture =
        utils::createTexture2D(envCubemapSize, envCubemapSize, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, envCubemapSize, envCubemapSize);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture.get(), 0);

    brdfShader->use();
    const ScreenQuad screenQuad;
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return brdfLUTTexture;
}

utils::TextureHandle PbrCubemapTexture::createCubemapAndCopyDataFromFirstLayerOf(utils::TextureHandle cubemap, unsigned cubemapSize)
{
    PUSH_DEBUG_GROUP(COPY_CUBEMAP_BASE_LAYER);

    utils::TextureHandle dstCubemap = utils::createCubemap(cubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    glCopyImageSubData(cubemap.get(), GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, dstCubemap.get(), GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, cubemapSize, cubemapSize,
                       6);

    POP_DEBUG_GROUP();
    return dstCubemap;
}
}  // namespace spark