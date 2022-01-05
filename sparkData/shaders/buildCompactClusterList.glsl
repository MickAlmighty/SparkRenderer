#type compute
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

layout(std430) buffer ActiveClusters
{
    uint activeClusters[];
};

layout(std430) buffer ActiveClustersCount
{
    uint globalActiveClusterCount;
};

layout(std430) buffer ActiveClusterIndices
{
    uint activeClusterIndices[];
};

uint calculateIndex()
{
    const uvec2 tilesCount = uvec2(64, 64);
    const uint screenSliceOffset = tilesCount.x * tilesCount.y * gl_GlobalInvocationID.z;
    const uint onScreenSliceIndex = gl_GlobalInvocationID.x * tilesCount.y + gl_GlobalInvocationID.y;
    return screenSliceOffset + onScreenSliceIndex;
}

shared uint localClusterIndices[256];
shared uint localActiveClusterCount;
shared uint globalActiveClusterCountIdx;

void main()
{
    if (gl_LocalInvocationIndex == 0)
    {
        localActiveClusterCount = 0;
        globalActiveClusterCountIdx = 0;
    }
    localClusterIndices[gl_LocalInvocationIndex] = 0;

    barrier();
    const uint clusterIndex = calculateIndex();
    if (activeClusters[clusterIndex] != 0)
    {
        activeClusters[clusterIndex] = 0;
        const uint localIdx = atomicAdd(localActiveClusterCount, 1);
        localClusterIndices[localIdx] = clusterIndex;
    }

    barrier();
    if (gl_LocalInvocationIndex == 0 && localActiveClusterCount != 0)
    {
        globalActiveClusterCountIdx = atomicAdd(globalActiveClusterCount, localActiveClusterCount);
    }

    barrier();
    if (gl_LocalInvocationIndex < localActiveClusterCount)
    {
        activeClusterIndices[globalActiveClusterCountIdx + gl_LocalInvocationIndex] = localClusterIndices[gl_LocalInvocationIndex];
    }
}