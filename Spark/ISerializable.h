#ifndef ISERIALIZABLE_H
#define ISERIALIZABLE_H

#include <json/value.h>

enum class SerializableType
{
	SGameObject = 1,
	SModelMesh,
	SCamera,
	SMeshPlane,
	STerrainGenerator
};


class ISerializable
{
public:
	virtual SerializableType getSerializableType() = 0;
	virtual Json::Value serialize() = 0;
	virtual void deserialize(Json::Value& root) = 0;
	virtual ~ISerializable() = default;
};
#endif