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


class ISerializable
{
public:
	virtual SerializableType getSerializableType() = 0;
	virtual Json::Value serialize() = 0;
	virtual void deserialize(Json::Value& root) = 0;
	virtual ~ISerializable() = default;
};

}
#endif