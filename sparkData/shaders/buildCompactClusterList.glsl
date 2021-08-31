//#type compute
#version 450
layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

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
    uint screenSliceOffset = tilesCount.x * tilesCount.y * gl_GlobalInvocationID.z;
    uint onScreenSliceIndex = gl_GlobalInvocationID.x * tilesCount.y + gl_GlobalInvocationID.y;
    return screenSliceOffset + onScreenSliceIndex;
}

void main()
{
    uint clusterIndex = calculateIndex();
    if (activeClusters[clusterIndex] != 0)
    {
        uint idx = atomicAdd(globalActiveClusterCount, 1);
        activeClusterIndices[idx] = clusterIndex;
        activeClusters[clusterIndex] = 0;
    }
}