#pragma once

#include <vector>

namespace spark
{
struct AttributeDescriptor final
{
    unsigned int location{0};
    unsigned int components{1};  // 1 - 4
    unsigned int stride{};       // in bytes
};

struct VertexAttribute final
{
    AttributeDescriptor descriptor{};
    std::vector<uint8_t> bytes{};

    bool operator<(const VertexAttribute& attribute) const
    {
        return descriptor.location < attribute.descriptor.location;
    }

    template<typename T>
    static VertexAttribute createVertexShaderAttributeInfo(unsigned int location, unsigned int components, std::vector<T> vertexAttributeData)
    {
        unsigned int elemSize = sizeof(T);

        VertexAttribute attribute;
        attribute.descriptor.location = location;
        attribute.descriptor.components = components;
        attribute.descriptor.stride = elemSize;
        attribute.bytes.resize(elemSize * vertexAttributeData.size());
        std::memcpy(attribute.bytes.data(), vertexAttributeData.data(), vertexAttributeData.size() * elemSize);

        return attribute;
    }
};

}  // namespace spark