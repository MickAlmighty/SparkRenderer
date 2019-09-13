#pragma once
#include <json/value.h>

enum class SerializableType
{
	GameObjectSerializable = 1
};


class ISerializable
{
public:
	virtual SerializableType getSerializableType() = 0;
	virtual Json::Value serialize(Json::Value& root) = 0;
	virtual void deserialize(Json::Value& root) = 0;
	virtual ~ISerializable() = default;
};
