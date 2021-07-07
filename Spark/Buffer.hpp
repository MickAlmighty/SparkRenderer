#pragma once

#include <algorithm>
#include <set>
#include <vector>

#include <glad/glad.h>

template<GLenum BUFFER_TYPE>
class Buffer
{
    public:
    static inline std::set<uint32_t> bindings{};

    GLuint ID{0};
    GLint binding{-1};
    GLsizei size{0};

    Buffer(std::size_t sizeInBytes = 0);
    Buffer(const Buffer& buffer) = delete;
    Buffer(Buffer&& buffer) = delete;
    Buffer& operator=(const Buffer& buffer) = delete;
    Buffer& operator=(Buffer&& buffer) = delete;
    ~Buffer();

    void bind() const;
    static void unbind();

    template<typename T>
    void updateData(const std::vector<T>& buffer);

    template<typename T>
    void updateSubData(size_t offsetFromBeginning, const std::vector<T>& buffer);

    template<typename T, size_t Size>
    void updateData(const std::array<T, Size>& buffer);

    template<typename T, size_t Size>
    void updateSubData(size_t offsetFromBeginning, const std::array<T, Size>& buffer);

    void resizeBuffer(size_t sizeInBytes);
    // this method sets value 0 for all bytes in the buffer
    void clearData() const;

    private:
    void genBuffer(size_t sizeInBytes = 0);
    void cleanup();
    void getBinding();
    void freeBinding();
};

template<GLenum BUFFER_TYPE>
Buffer<BUFFER_TYPE>::Buffer(std::size_t sizeInBytes)
{
    genBuffer(sizeInBytes);
}

template<GLenum BUFFER_TYPE>
Buffer<BUFFER_TYPE>::~Buffer()
{
    if(ID != 0)
    {
        cleanup();
    }
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::bind() const
{
    glBindBuffer(BUFFER_TYPE, ID);
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::unbind()
{
    glBindBuffer(BUFFER_TYPE, 0);
}

template<GLenum BUFFER_TYPE>
template<typename T>
void Buffer<BUFFER_TYPE>::updateData(const std::vector<T>& buffer)
{
    const size_t vectorSize = buffer.size() * sizeof(T);
    if(vectorSize < size || vectorSize > size)
    {
        // SPARK_WARN("Trying to update SSBO with a vector with too large size! SSBO size: {}, vector size: {}. Buffer will be resized and update
        // will be processed!", size, vectorSize);
        size = static_cast<GLsizei>(vectorSize);
        glNamedBufferData(ID, vectorSize, buffer.data(), GL_DYNAMIC_DRAW);
    }
    else
    {
        glNamedBufferSubData(ID, 0, vectorSize, buffer.data());
    }
}

template<GLenum BUFFER_TYPE>
template<typename T>
void Buffer<BUFFER_TYPE>::updateSubData(size_t offsetFromBeginning, const std::vector<T>& buffer)
{
    const size_t vectorSize = buffer.size() * sizeof(T);
    const GLintptr offset = static_cast<GLintptr>(offsetFromBeginning);

    if(offset > size)
    {
        return;
    }
    if(offset + vectorSize > size)
    {
        return;
    }

    glNamedBufferSubData(ID, offset, vectorSize, buffer.data());
}

template<GLenum BUFFER_TYPE>
template<typename T, size_t Size>
void Buffer<BUFFER_TYPE>::updateData(const std::array<T, Size>& buffer)
{
    const size_t vectorSize = Size * sizeof(T);
    if(vectorSize < size || vectorSize > size)
    {
        // SPARK_WARN("Trying to update SSBO with a vector with too large size! SSBO size: {}, vector size: {}. Buffer will be resized and update
        // will be processed!", size, vectorSize);
        size = static_cast<GLsizei>(vectorSize);
        glNamedBufferData(ID, vectorSize, buffer.data(), GL_DYNAMIC_DRAW);
    }
    else
    {
        glNamedBufferSubData(ID, 0, vectorSize, buffer.data());
    }
}

template<GLenum BUFFER_TYPE>
template<typename T, size_t Size>
void Buffer<BUFFER_TYPE>::updateSubData(size_t offsetFromBeginning, const std::array<T, Size>& buffer)
{
    const size_t vectorSize = Size * sizeof(T);
    const GLintptr offset = static_cast<GLintptr>(offsetFromBeginning);

    if(offset > size)
    {
        return;
    }
    if(offset + vectorSize > size)
    {
        return;
    }

    glNamedBufferSubData(ID, offset, vectorSize, buffer.data());
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::resizeBuffer(size_t sizeInBytes)
{
    glNamedBufferData(ID, sizeInBytes, nullptr, GL_DYNAMIC_DRAW);
    size = static_cast<GLsizei>(sizeInBytes);
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::clearData() const
{
    glClearNamedBufferData(ID, GL_R32F, GL_RED, GL_FLOAT, nullptr);
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::genBuffer(size_t sizeInBytes)
{
    if (ID != 0)
    {
        cleanup();
    }

    size = static_cast<GLsizei>(sizeInBytes);
    glGenBuffers(1, &ID);
    glBindBuffer(BUFFER_TYPE, ID);
    glBufferData(BUFFER_TYPE, size, nullptr, GL_DYNAMIC_DRAW);
    glBindBuffer(BUFFER_TYPE, 0);
    getBinding();
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::cleanup()
{
    glDeleteBuffers(1, &ID);
    ID = 0;
    freeBinding();
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::getBinding()
{
    constexpr auto findFreeBindingBetweenAdjacentBindings = [](const uint32_t& binding1, const uint32_t& binding2) {
        constexpr auto allowedDistanceBetweenBindings = 1;
        return (binding2 - binding1) > allowedDistanceBetweenBindings;
    };

    const auto it = std::adjacent_find(bindings.begin(), bindings.end(), findFreeBindingBetweenAdjacentBindings);
    if(it != bindings.end())
    {
        binding = *it + 1;
        bindings.insert(binding);
    }
    else
    {
        if(bindings.empty())
        {
            binding = 0;
            bindings.insert(binding);
        }
        else
        {
            binding = *std::prev(bindings.end()) + 1;
            bindings.insert(binding);
        }
    }
}

template<GLenum BUFFER_TYPE>
void Buffer<BUFFER_TYPE>::freeBinding()
{
    const auto it = bindings.find(binding);
    if(it != bindings.end())
    {
        bindings.erase(it);
        binding = -1;
    }
}

using SSBO = Buffer<GL_SHADER_STORAGE_BUFFER>;
using UniformBuffer = Buffer<GL_UNIFORM_BUFFER>;
using ElementArrayBuffer = Buffer<GL_ELEMENT_ARRAY_BUFFER>;
using VertexBuffer = Buffer<GL_ARRAY_BUFFER>;