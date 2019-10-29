#ifndef ISERIALIZABLE_H
#define ISERIALIZABLE_H

#include <json/value.h>

namespace spark {

enum class SerializableType : uint16_t
{
	SGameObject = 1,
	SModelMesh,
	SCamera,
	SMeshPlane,
	STerrainGenerator,
	SActorAI
};

}
#endif