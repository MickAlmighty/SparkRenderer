#include "PbrCubemapTexture.hpp"

#include "CommonUtils.h"
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

PbrCubemapTexture::~PbrCubemapTexture()
{
    std::array<GLuint, 3> textures = {cubemap, irradianceCubemap, prefilteredCubemap};
    glDeleteTextures(3, textures.data());
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

    const GLuint envCubemap = createEnvironmentCubemapWithMipmapChain(captureFBO, hdrTexture, cubemapSize, cube, equirectangularToCubemapShader);
    this->irradianceCubemap = createIrradianceCubemap(captureFBO, envCubemap, cube, irradianceShader);
    this->prefilteredCubemap = createPreFilteredCubemap(captureFBO, envCubemap, cubemapSize, cube, prefilterShader, resampleCubemapShader);

    // creating cubemap in the line below is meant to reduce memory usage by getting rid of mipmaps
    this->cubemap = createCubemapAndCopyDataFromFirstLayerOf(envCubemap, cubemapSize);
    // this->brdfLUTTexture = createBrdfLookupTexture(captureFBO, 1024);

    glDeleteFramebuffers(1, &captureFBO);
    glDeleteTextures(1, &envCubemap);
}

GLuint PbrCubemapTexture::createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned size, Cube& cube,
                                                                  const std::shared_ptr<resources::Shader>& equirectangularToCubemapShader)
{
    PUSH_DEBUG_GROUP(EQUIRECTANGULAR_TO_CUBEMAP_WITH_MIPS);

    GLuint envCubemap{};
    utils::createCubemap(envCubemap, size, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    glViewport(0, 0, size, size);

    equirectangularToCubemapShader->use();
    glBindTextureUnit(0, equirectangularTexture);

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, envCubemap, 0);
    cube.draw();

    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    POP_DEBUG_GROUP();
    return envCubemap;
}

GLuint PbrCubemapTexture::createIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube,
                                                  const std::shared_ptr<resources::Shader>& irradianceShader)
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP);

    GLuint irradianceMap{};
    utils::createCubemap(irradianceMap, 32, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, 32, 32);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap);

    glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, irradianceMap, 0);
    cube.draw();

    POP_DEBUG_GROUP();

    return irradianceMap;
}

GLuint PbrCubemapTexture::createPreFilteredCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube,
                                                   const std::shared_ptr<resources::Shader>& prefilterShader,
                                                   const std::shared_ptr<resources::Shader>& resampleCubemapShader)
{
    PUSH_DEBUG_GROUP(PREFILTER_CUBEMAP);

    const unsigned int prefilteredMapSize = 128;
    GLuint prefilteredMap{};
    utils::createCubemap(prefilteredMap, prefilteredMapSize, GL_R11F_G11F_B10F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    {
        resampleCubemapShader->use();
        glBindTextureUnit(0, environmentCubemap);

        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilteredMap, 0);

        glViewport(0, 0, prefilteredMapSize, prefilteredMapSize);
        cube.draw();
    }

    const GLuint maxMipLevels = 5;
    prefilterShader->use();
    glBindTextureUnit(0, environmentCubemap);
    prefilterShader->setFloat("textureSize", static_cast<float>(envCubemapSize));

    for(unsigned int mip = 1; mip < maxMipLevels; ++mip)
    {
        const auto mipSize = static_cast<unsigned int>(prefilteredMapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipSize, mipSize);
        glNamedFramebufferTexture(framebuffer, GL_COLOR_ATTACHMENT0, prefilteredMap, mip);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);

        cube.draw();
    }

    POP_DEBUG_GROUP();

    return prefilteredMap;
}

GLuint PbrCubemapTexture::createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize,
                                                  const std::shared_ptr<resources::Shader>& brdfShader)
{
    PUSH_DEBUG_GROUP(BRDF_LOOKUP_TABLE);

    GLuint brdfLUTTexture{};
    utils::createTexture2D(brdfLUTTexture, envCubemapSize, envCubemapSize, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, envCubemapSize, envCubemapSize);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    brdfShader->use();
    const ScreenQuad screenQuad;
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return brdfLUTTexture;
}

GLuint PbrCubemapTexture::createCubemapAndCopyDataFromFirstLayerOf(GLuint cubemap, unsigned cubemapSize)
{
    PUSH_DEBUG_GROUP(COPY_CUBEMAP_BASE_LAYER);

    GLuint dstCubemap{};
    utils::createCubemap(dstCubemap, cubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    glCopyImageSubData(cubemap, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, dstCubemap, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, cubemapSize, cubemapSize, 6);

    POP_DEBUG_GROUP();
    return dstCubemap;
}
}  // namespace spark