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

class VertexAttribute final
{
    public:
    template<typename T>
    VertexAttribute(unsigned int location, unsigned int components, std::vector<T> vertexAttributeData)
    {
        unsigned int elemSize = sizeof(T);
        descriptor.location = location;
        descriptor.components = components;
        descriptor.stride = elemSize;
        bytes.resize(elemSize * vertexAttributeData.size());
        std::memcpy(bytes.data(), vertexAttributeData.data(), vertexAttributeData.size() * elemSize);
    }

    bool operator<(const VertexAttribute& attribute) const
    {
        return descriptor.location < attribute.descriptor.location;
    }

    AttributeDescriptor descriptor{};
    std::vector<uint8_t> bytes{};
};

}  // namespace spark