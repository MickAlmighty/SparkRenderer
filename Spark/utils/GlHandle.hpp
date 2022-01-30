#pragma once

#include <memory>

#include "glad_glfw3.h"

namespace spark::utils
{
struct TextureDeleter final
{
    void operator()(const GLuint* pointerWithHandle) const
    {
        const auto handle = static_cast<GLuint>(reinterpret_cast<GLuint64>(pointerWithHandle));
        if(handle != 0)
        {
            glDeleteTextures(1, &handle);
        }
    }
};

template<typename Deleter>
class GlHandle
{
    public:
    GlHandle(GLuint handle = 0)
    {
        if(handle > 0)
        {
            GLuint* handleAsPointerValue{reinterpret_cast<GLuint*>(static_cast<GLuint64>(handle))};
            handleHolder = std::shared_ptr<GLuint>(handleAsPointerValue, Deleter());
        }
    }

    GLuint get() const
    {
        return static_cast<GLuint>(reinterpret_cast<GLuint64>(handleHolder.get()));
    }

    size_t use_count() const
    {
        return handleHolder.use_count();
    }

    private:
    std::shared_ptr<GLuint> handleHolder{nullptr};
};

using TextureHandle = GlHandle<TextureDeleter>;
}  // namespace spark::utils