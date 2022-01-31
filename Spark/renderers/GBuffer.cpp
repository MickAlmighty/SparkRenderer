#include "GBuffer.hpp"

#include "utils/CommonUtils.h"
#include "Logging.h"
#include "Shader.h"
#include "Spark.h"

namespace spark::renderers
{
GBuffer::GBuffer(unsigned int width, unsigned int height) : w(width), h(height)
{
    pbrGeometryBufferShader = Spark::get().getResourceLibrary().getResourceByName<resources::Shader>("pbrGeometryBuffer.glsl");
    createFrameBuffersAndTextures();
}

void GBuffer::fill(const std::map<ShaderType, std::deque<RenderingRequest>>& renderingQueues, const UniformBuffer& cameraUbo)
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

    if(const auto it = renderingQueues.find(ShaderType::PBR); it != renderingQueues.cend())
    {
        for(auto& request : it->second)
        {
            request.mesh->draw(pbrGeometryBufferShader, request.model);
        }
    }

    POP_DEBUG_GROUP();
}

void GBuffer::fill(const std::map<ShaderType, std::deque<RenderingRequest>>& renderingQueues,
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
    if(const auto it = renderingQueues.find(ShaderType::PBR); it != renderingQueues.cend())
    {
        for(auto& request : it->second)
        {
            if(filter(request))
            {
                request.mesh->draw(pbrGeometryBufferShader, request.model);
            }
        }
    }

    POP_DEBUG_GROUP();
}

GBuffer::~GBuffer()
{
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
    colorTexture = utils::createTexture2D(w, h, GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    normalsTexture = utils::createTexture2D(w, h, GL_RG16F, GL_RG, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);
    roughnessMetalnessTexture = utils::createTexture2D(w, h, GL_RG, GL_RG, GL_UNSIGNED_BYTE, GL_CLAMP_TO_EDGE, GL_LINEAR);
    depthTexture = utils::createTexture2D(w, h, GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_FLOAT, GL_CLAMP_TO_EDGE, GL_LINEAR);

    utils::recreateFramebuffer(framebuffer, {colorTexture.get(), normalsTexture.get(), roughnessMetalnessTexture.get()});
    utils::bindDepthTexture(framebuffer, depthTexture.get());
}
}  // namespace spark::renderers