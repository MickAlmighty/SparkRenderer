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
class SharedGlHandle;

template<typename Deleter>
class UniqueGlHandle
{
    public:
    UniqueGlHandle(GLuint handle = 0) noexcept
    {
        if(handle > 0)
        {
            GLuint* handleAsPointerValue{reinterpret_cast<GLuint*>(static_cast<GLuint64>(handle))};
            handleHolder = std::unique_ptr<GLuint, Deleter>(handleAsPointerValue);
        }
    }

    UniqueGlHandle(const UniqueGlHandle&) noexcept = delete;
    UniqueGlHandle(UniqueGlHandle&&) noexcept = default;
    UniqueGlHandle& operator=(const UniqueGlHandle&) noexcept = delete;
    UniqueGlHandle& operator=(UniqueGlHandle&&) noexcept = default;
    ~UniqueGlHandle() = default;

    GLuint get() const
    {
        return static_cast<GLuint>(reinterpret_cast<GLuint64>(handleHolder.get()));
    }

    private:
    std::unique_ptr<GLuint, Deleter> handleHolder{nullptr};

    friend class SharedGlHandle<Deleter>;
};

template<typename Deleter>
class SharedGlHandle
{
    public:
    SharedGlHandle(GLuint handle = 0) noexcept
    {
        if(handle > 0)
        {
            GLuint* handleAsPointerValue{reinterpret_cast<GLuint*>(static_cast<GLuint64>(handle))};
            handleHolder = std::shared_ptr<GLuint>(handleAsPointerValue, Deleter());
        }
    }

    SharedGlHandle(UniqueGlHandle<Deleter>&& uniqueHandle) noexcept
    {
        this->handleHolder = std::move(uniqueHandle.handleHolder);
    }

    SharedGlHandle& operator=(UniqueGlHandle<Deleter>&& uniqueHandle) noexcept
    {
        this->handleHolder = std::move(uniqueHandle.handleHolder);
        return *this;
    }

    SharedGlHandle(const SharedGlHandle&) noexcept = default;
    SharedGlHandle(SharedGlHandle&&) noexcept = default;
    SharedGlHandle& operator=(const SharedGlHandle&) noexcept = default;
    SharedGlHandle& operator=(SharedGlHandle&&) noexcept = default;
    ~SharedGlHandle() = default;

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

using TextureHandle = SharedGlHandle<TextureDeleter>;
using UniqueTextureHandle = UniqueGlHandle<TextureDeleter>;
}  // namespace spark::utils