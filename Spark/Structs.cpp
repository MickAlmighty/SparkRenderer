#include "Structs.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{
Texture::Texture(GLuint id, const std::string& path)
{
    this->path = path;
    this->ID = id;
}

void Texture::setPath(const std::string path)
{
    this->path = path;
    //this->ID = ResourceManager::getInstance()->getTextureId(path);
}

std::string Texture::getPath() const
{
    return path;
}

const std::string PbrCubemapTexture::getPath()
{
    return path;
}

PbrCubemapTexture::PbrCubemapTexture(GLuint hdrTexture, const std::string& path, unsigned size) : path(path)
{
    equirectangularToCubemapShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("equirectangularToCubemap.glsl");
    irradianceShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("irradiance.glsl");
    prefilterShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("prefilter.glsl");
    brdfShader = Spark::getResourceLibrary()->getResourceByNameWithOptLoad<resources::Shader>("brdf.glsl");
    setup(hdrTexture, size);
}

PbrCubemapTexture::~PbrCubemapTexture()
{
    GLuint textures[] = {cubemap, irradianceCubemap, prefilteredCubemap, brdfLUTTexture};
    glDeleteTextures(4, textures);
}

void PbrCubemapTexture::setup(GLuint hdrTexture, unsigned size)
{
    const glm::mat4 captureProjection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    glm::mat4 captureViews[] = {glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
                                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                                glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};
    Cube cube = Cube();

    const unsigned int cubemapSize = size;
    GLuint captureFBO;
    glGenFramebuffers(1, &captureFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);
    const GLuint envCubemap = generateCubemap(size, false);

    equirectangularToCubemapShader->use();
    equirectangularToCubemapShader->setMat4("projection", captureProjection);
    glBindTextureUnit(0, hdrTexture);

    glViewport(0, 0, cubemapSize, cubemapSize);

    for(unsigned int i = 0; i < 6; ++i)
    {
        equirectangularToCubemapShader->setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cube.draw();
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    ///////// IRRADIANCE
    GLuint irradianceMap = generateCubemap(32, false);

    glViewport(0, 0, 32, 32);

    irradianceShader->use();
    glBindTextureUnit(0, envCubemap);
    irradianceShader->setMat4("projection", captureProjection);

    for(unsigned int i = 0; i < 6; ++i)
    {
        irradianceShader->setMat4("view", captureViews[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        cube.draw();
    }

    ////// PREFILTER
    const unsigned int prefilterMapSize = 128;
    const GLuint prefilterMap = generateCubemap(128, true);

    prefilterShader->use();
    glBindTextureUnit(0, envCubemap);
    prefilterShader->setMat4("projection", captureProjection);
    prefilterShader->setFloat("textureSize", static_cast<float>(cubemapSize));

    const GLuint maxMipLevels = 5;
    for(unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        const unsigned int mipWidth = static_cast<unsigned int>(prefilterMapSize * std::pow(0.5, mip));
        const unsigned int mipHeight = static_cast<unsigned int>(prefilterMapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipWidth, mipHeight);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);
        for(unsigned int i = 0; i < 6; ++i)
        {
            prefilterShader->setMat4("view", captureViews[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilterMap, mip);

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            cube.draw();
        }
    }

    //////// BRDF

    unsigned int brdfLUTTexture;
    glGenTextures(1, &brdfLUTTexture);

    glBindTexture(GL_TEXTURE_2D, brdfLUTTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, cubemapSize, cubemapSize, 0, GL_RG, GL_FLOAT, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    glViewport(0, 0, cubemapSize, cubemapSize);

    brdfShader->use();
    ScreenQuad screenQuad;
    screenQuad.setup();
    screenQuad.draw();

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &captureFBO);
    this->cubemap = envCubemap;
    this->irradianceCubemap = irradianceMap;
    this->prefilteredCubemap = prefilterMap;
    this->brdfLUTTexture = brdfLUTTexture;
}

GLuint PbrCubemapTexture::generateCubemap(unsigned texSize, bool mipmaps) const
{
    GLuint cubemap;
    glGenTextures(1, &cubemap);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
    for(unsigned int i = 0; i < 6; ++i)
    {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB16F, texSize, texSize, 0, GL_RGB, GL_FLOAT, nullptr);
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    if(mipmaps)
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
    }
    else
    {
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    }
    return cubemap;
}

void PbrCubemapTexture::generateCubemapMipMaps()
{
    glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::InitializationVariables>("InitializationVariables")
        .constructor()(rttr::policy::ctor::as_object)
        .property("width", &spark::InitializationVariables::width)
        .property("height", &spark::InitializationVariables::height)
        .property("pathToModels", &spark::InitializationVariables::pathToModels)
        .property("pathToResources", &spark::InitializationVariables::pathToResources);

    rttr::registration::class_<spark::Transform>("Transform")
        .constructor()(rttr::policy::ctor::as_object)
        .property("local", &spark::Transform::local)
        .property("world", &spark::Transform::world);

    rttr::registration::class_<spark::Texture>("Texture")
        .constructor()(rttr::policy::ctor::as_object)
        .property("path", &spark::Texture::getPath, &spark::Texture::setPath, rttr::registration::public_access);
}