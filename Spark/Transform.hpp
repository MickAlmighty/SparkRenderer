#pragma once

#include "LocalTransform.h"
#include "WorldTransform.h"

namespace spark
{
class Transform final
{
    public:
    Transform();
    LocalTransform local;
    WorldTransform world;
};
}  // namespace spark
