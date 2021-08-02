#include "GBuffer.h"

#include "CommonUtils.h"
#include "Logging.h"
#include "Shader.h"
#include "Spark.h"

namespace spark
{

GBuffer::GBuffer(unsigned int width, unsigned int height)
{
    pbrGeometryBufferShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("pbrGeometryBuffer.glsl");
    resize(width, height);
}

void GBuffer::fill(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue, const UniformBuffer& cameraUbo)
{
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0, 0, 0, 0);
    glClearDepth(0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    pbrGeometryBufferShader->use();
    pbrGeometryBufferShader->bindUniformBuffer("Camera", cameraUbo);
    for(auto& request : renderQueue[ShaderType::PBR])
    {
        request.mesh->draw(pbrGeometryBufferShader, request.model);
    }

    POP_DEBUG_GROUP();
}

void GBuffer::fill(std::map<ShaderType, std::deque<RenderingRequest>>& renderQueue,
                   const std::function<bool(const RenderingRequest& request)>& filter, const UniformBuffer& cameraUbo)
{
    PUSH_DEBUG_GROUP(RENDER_TO_MAIN_FRAMEBUFFER_FILTERED);

    glViewport(0, 0, w, h);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(0.0);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);

    pbrGeometryBufferShader->use();
    pbrGeometryBufferShader->bindUniformBuffer("Camera", cameraUbo);
    for(auto& request : renderQueue[ShaderType::PBR])
    {
        if(filter(request))
        {
            request.mesh->draw(pbrGeometryBufferShader, request.model);
        }
    }

    POP_DEBUG_GROUP();
}

GBuffer::~GBuffer()
{
    GLuint textures[4] = {colorTexture, normalsTexture, roughnessMetalnessTexture, depthTexture};
    glDeleteTextures(4, textures);
    colorTexture = normalsTexture = roughnessMetalnessTexture = depthTexture = 0;

    glDeleteFramebuffers(1, &framebuffer);
    framebuffer = 0;
}

void GBuffer::resize(unsigned int width, unsigned int height)
{
    w = width;
    h = height;
    createFrameBuffersAndTextures();
}

void GBuffer::createFrameBuffersAndTextures()
{
    utils::recreateTexture2D(colorTexture, w, h, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(normalsTexture, w, h, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(roughnessMetalnessTexture, w, h, GL_RG, GL_RG, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    utils::recreateTexture2D(depthTexture, w, h, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(framebuffer, { colorTexture, normalsTexture, roughnessMetalnessTexture });
    utils::bindDepthTexture(framebuffer, depthTexture);
}
}  // namespace spark