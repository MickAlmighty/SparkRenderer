#include "Structs.h"

#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>

#include "CommonUtils.h"
#include "Logging.h"
#include "ResourceLibrary.h"
#include "Shader.h"
#include "Spark.h"
#include "Timer.h"

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
    // this->ID = ResourceManager::getInstance()->getTextureId(path);
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

    projection = glm::perspective(glm::radians(90.0f), 1.0f, 0.1f, 10.0f);
    viewMatrices = {glm::lookAt(glm::vec3(0.0f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                    glm::lookAt(glm::vec3(0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
                    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(0.0f, 0.0f, -1.0f)),
                    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, -1.0f, 0.0f)),
                    glm::lookAt(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, -1.0f, 0.0f))};

    setup(hdrTexture, size);
}

PbrCubemapTexture::~PbrCubemapTexture()
{
    GLuint textures[] = {cubemap, irradianceCubemap, prefilteredCubemap, brdfLUTTexture};
    glDeleteTextures(4, textures);
}

void PbrCubemapTexture::setup(GLuint hdrTexture, unsigned size)
{
    Timer t("HDR PBR cubemap generation time");
    Cube cube = Cube();

    const unsigned int cubemapSize = size;
    GLuint captureFBO;
    glGenFramebuffers(1, &captureFBO);
    glBindFramebuffer(GL_FRAMEBUFFER, captureFBO);

    const GLuint envCubemap = createEnvironmentCubemapWithMipmapChain(captureFBO, hdrTexture, cubemapSize, cube);

    // creating cubemap in the line below is meant to reduce memory usage by getting rid of mipmaps
    this->cubemap = createCubemapAndCopyDataFromFirstLayerOf(envCubemap, cubemapSize);
    this->irradianceCubemap = createIrradianceCubemap(captureFBO, envCubemap, cube);
    this->prefilteredCubemap = createPreFilteredCubemap(captureFBO, envCubemap, cubemapSize, cube);
    // this->brdfLUTTexture = createBrdfLookupTexture(captureFBO, 1024);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &captureFBO);
    glDeleteTextures(1, &envCubemap);
}

GLuint PbrCubemapTexture::createEnvironmentCubemapWithMipmapChain(GLuint framebuffer, GLuint equirectangularTexture, unsigned size, Cube& cube) const
{
    PUSH_DEBUG_GROUP(EQUIRECTANGULAR_TO_CUBEMAP_WITH_MIPS);

    GLuint envCubemap{};
    utils::createCubemap(envCubemap, size, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, size, size);

    equirectangularToCubemapShader->use();
    equirectangularToCubemapShader->setMat4("projection", projection);
    glBindTextureUnit(0, equirectangularTexture);

    for(unsigned int i = 0; i < 6; ++i)
    {
        equirectangularToCubemapShader->setMat4("view", viewMatrices[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, envCubemap, 0);
        cube.draw();
    }

    glBindTexture(GL_TEXTURE_CUBE_MAP, envCubemap);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_CUBE_MAP);

    POP_DEBUG_GROUP();
    return envCubemap;
}

GLuint PbrCubemapTexture::createIrradianceCubemap(GLuint framebuffer, GLuint environmentCubemap, Cube& cube) const
{
    PUSH_DEBUG_GROUP(IRRADIANCE_CUBEMAP);

    GLuint irradianceMap{};
    utils::createCubemap(irradianceMap, 32, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, 32, 32);

    irradianceShader->use();
    glBindTextureUnit(0, environmentCubemap);
    irradianceShader->setMat4("projection", projection);

    for(unsigned int i = 0; i < 6; ++i)
    {
        irradianceShader->setMat4("view", viewMatrices[i]);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, irradianceMap, 0);

        cube.draw();
    }

    POP_DEBUG_GROUP();

    return irradianceMap;
}

GLuint PbrCubemapTexture::createPreFilteredCubemap(GLuint framebuffer, GLuint environmentCubemap, unsigned envCubemapSize, Cube& cube) const
{
    PUSH_DEBUG_GROUP(PREFILTER_CUBEMAP);

    const unsigned int prefilteredMapSize = 128;
    GLuint prefilteredMap{};
    utils::createCubemap(prefilteredMap, prefilteredMapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR, true);

    prefilterShader->use();
    glBindTextureUnit(0, environmentCubemap);
    prefilterShader->setMat4("projection", projection);
    prefilterShader->setFloat("textureSize", static_cast<float>(envCubemapSize));

    const GLuint maxMipLevels = 5;
    for(unsigned int mip = 0; mip < maxMipLevels; ++mip)
    {
        const auto mipWidth = static_cast<unsigned int>(prefilteredMapSize * std::pow(0.5, mip));
        const auto mipHeight = static_cast<unsigned int>(prefilteredMapSize * std::pow(0.5, mip));
        glViewport(0, 0, mipWidth, mipHeight);

        const float roughness = static_cast<float>(mip) / static_cast<float>(maxMipLevels - 1);
        prefilterShader->setFloat("roughness", roughness);
        for(unsigned int i = 0; i < 6; ++i)
        {
            prefilterShader->setMat4("view", viewMatrices[i]);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, prefilteredMap, mip);

            cube.draw();
        }
    }

    POP_DEBUG_GROUP();

    return prefilteredMap;
}

GLuint PbrCubemapTexture::createBrdfLookupTexture(GLuint framebuffer, unsigned int envCubemapSize) const
{
    PUSH_DEBUG_GROUP(BRDF_LOOKUP_TABLE);

    GLuint brdfLUTTexture{};
    utils::createTexture2D(brdfLUTTexture, envCubemapSize, envCubemapSize, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    glViewport(0, 0, envCubemapSize, envCubemapSize);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, brdfLUTTexture, 0);

    brdfShader->use();
    ScreenQuad screenQuad;
    screenQuad.setup();
    screenQuad.draw();

    POP_DEBUG_GROUP();

    return brdfLUTTexture;
}

GLuint PbrCubemapTexture::createCubemapAndCopyDataFromFirstLayerOf(GLuint cubemap, unsigned cubemapSize) const
{
    PUSH_DEBUG_GROUP(COPY_CUBEMAP_BASE_LAYER);

    GLuint dstCubemap{};
    utils::createCubemap(dstCubemap, cubemapSize, GL_RGB16F, GL_RGB, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    glCopyImageSubData(cubemap, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, dstCubemap, GL_TEXTURE_CUBE_MAP, 0, 0, 0, 0, cubemapSize, cubemapSize, 6);

    POP_DEBUG_GROUP();
    return dstCubemap;
}

Cube::Cube()
{
    float vertices[288] = {
        // back face
        -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left
        1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f,   // bottom-right
        1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,    // top-right

        1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,    // top-right
        -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,   // top-left
        -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left

        // front face
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,    // top-right
        1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,   // bottom-right
        -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom-left

        -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,   // top-left
        1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,    // top-right
        -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom-left

        // left face
        -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-right
        -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-left
        -1.0f, 1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   // top-left

        -1.0f, -1.0f, -1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-left
        -1.0f, 1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-right
        -1.0f, -1.0f, 1.0f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // bottom-right

        // right face
        1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-right
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-left
        1.0f, 1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f,   // top-right

        1.0f, -1.0f, -1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-right
        1.0f, -1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,   // bottom-left
        1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,    // top-left

        // bottom face
        1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f,   // top-left
        -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // top-right
        1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,    // bottom-left

        1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,    // bottom-left
        -1.0f, -1.0f, -1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f,  // top-right
        -1.0f, -1.0f, 1.0f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f,   // bottom-right

        // top face
        1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,    // bottom-right
        -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,  // top-left
        1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f,   // top-right

        1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f,   // bottom-right
        -1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,  // bottom-left
        -1.0f, 1.0f, -1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f  // top-left

    };

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    // fill buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    // link vertex attributes
    glBindVertexArray(vao);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}
}  // namespace spark

RTTR_REGISTRATION
{
    rttr::registration::class_<spark::InitializationVariables>("InitializationVariables")
        .constructor()(rttr::policy::ctor::as_object)
        .property("width", &spark::InitializationVariables::width)
        .property("height", &spark::InitializationVariables::height)
        .property("pathToModels", &spark::InitializationVariables::pathToModels)
        .property("pathToResources", &spark::InitializationVariables::pathToResources)
        .property("vsync", &spark::InitializationVariables::vsync);

    rttr::registration::class_<spark::Transform>("Transform")
        .constructor()(rttr::policy::ctor::as_object)
        .property("local", &spark::Transform::local)
        .property("world", &spark::Transform::world);

    rttr::registration::class_<spark::Texture>("Texture")
        .constructor()(rttr::policy::ctor::as_object)
        .property("path", &spark::Texture::getPath, &spark::Texture::setPath, rttr::registration::public_access);
}