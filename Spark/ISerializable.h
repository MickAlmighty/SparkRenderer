#pragma once
#include <json/value.h>

enum class SerializableType
{
	SGameObject = 1,
	SModelMesh
};


class ISerializable
{
public:
	virtual SerializableType getSerializableType() = 0;
	virtual Json::Value serialize() = 0;
	virtual void deserialize(Json::Value& root) = 0;
	virtual ~ISerializable() = default;
};
