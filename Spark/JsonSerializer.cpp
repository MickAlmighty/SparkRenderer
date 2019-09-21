#include <JsonSerializer.h>
#include <iostream>
#include <fstream>
#include <json/reader.h>
#include <json/writer.h>
#include <GameObject.h>
#include <ModelMesh.h>
#include <filesystem>
#include <ISerializable.h>

std::map<std::shared_ptr<ISerializable>, int> JsonSerializer::serializedObjects;

JsonSerializer::JsonSerializer()
{
}


JsonSerializer::~JsonSerializer()
{
}

std::shared_ptr<ISerializable> JsonSerializer::findSerializedObject(const int id)
{
	for(const auto& it : serializedObjects)
	{
		if(it.second == id)
		{
			return it.first;
		}
	}
	return nullptr;
}

int JsonSerializer::findId(const std::shared_ptr<ISerializable>& serializableObject)
{
	for (auto it : serializedObjects)
	{
		if (it.first == serializableObject)
		{
			return it.second;
		}
	}
	return -1;
}

void JsonSerializer::writeToFile(std::filesystem::path&& filePath, Json::Value&& root)
{
	Json::StreamWriterBuilder builder;
	std::ofstream file(filePath);
	Json::StreamWriter* writer = builder.newStreamWriter();
	writer->write(root, &file);
}

Json::Value JsonSerializer::readFromFile(std::filesystem::path&& filePath)
{
	Json::Value root;
	std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
	if (file.is_open())
	{
		auto size = file.tellg();
		char* data = new char[size];
		file.seekg(0, std::ios::beg);
		file.read(data, size);
		
		file.close();

		Json::CharReaderBuilder builder;
		Json::CharReader* reader = builder.newCharReader();

		std::string errors;
		reader->parse(data, data + size, &root, &errors);
		delete[] data;
	}
	file.close();
	return root;
}

Json::Value JsonSerializer::serialize(const std::shared_ptr<ISerializable> objToSerialize)
{
	Json::Value root;
	const int id = findId(objToSerialize);
	if(id != -1)
	{
		//if id != -1 means that this object has been serialized already
		root["id"] = id;
		root["SerializableType"] = static_cast<int>(objToSerialize->getSerializableType());
		return root;
	}
	
	counter++;
	serializedObjects.emplace(objToSerialize, counter);
	root["id"] = counter;
	root["SerializableType"] = static_cast<int>(objToSerialize->getSerializableType());
	root["object"] = objToSerialize->serialize();
	return root;
}



std::shared_ptr<ISerializable> JsonSerializer::deserialize(Json::Value& root)
{
	int id = root["id"].asInt();
	if(const auto obj = findSerializedObject(id); obj != nullptr)
	{
		return obj;
	}
	
	SerializableType type = static_cast<SerializableType>(root["SerializableType"].asInt());
	std::shared_ptr<ISerializable> deserialized;
	switch(type)
	{
	case SerializableType::SGameObject:
		deserialized = make<GameObject>();
		break;
	case SerializableType::SModelMesh:
		deserialized = make<ModelMesh>();
		break;
	default: 
		throw std::exception("Unsupported SerializableType encountered!");;
	}
	serializedObjects.emplace(deserialized, id);
	deserialized->deserialize(root["object"]);
	return deserialized;
}

void JsonSerializer::clearState()
{
	serializedObjects.clear();
	counter = 0;
}

Json::Value JsonSerializer::serializeVec2(glm::vec2 val)
{
	Json::Value root;
	root[0] = val.x;
	root[1] = val.y;
	return root;
}

glm::vec2 JsonSerializer::deserializeVec2(Json::Value& root)
{
	glm::vec2 val;
	val.x = root.get(Json::ArrayIndex(0), 0).asFloat();
	val.y = root.get(Json::ArrayIndex(1), 0).asFloat();
	return val;
}

Json::Value JsonSerializer::serializeVec3(glm::vec3 val)
{
	Json::Value root;
	root[0] = val.x;
	root[1] = val.y;
	root[2] = val.z;
	return root;
}

glm::vec3 JsonSerializer::deserializeVec3(Json::Value& root)
{
	glm::vec3 val;
	val.x = root.get(Json::ArrayIndex(0), 0).asFloat();
	val.y = root.get(Json::ArrayIndex(1), 0).asFloat();
	val.z = root.get(Json::ArrayIndex(2), 0).asFloat();
	return val;
}

Json::Value JsonSerializer::serializeVec4(glm::vec4 val)
{
	Json::Value root;
	root[0] = val.x;
	root[1] = val.y;
	root[2] = val.z;
	root[3] = val.w;
	return root;
}

glm::vec4 JsonSerializer::deserializeVec4(Json::Value& root)
{
	glm::vec4 val;
	val.x = root.get(Json::ArrayIndex(0), 0).asFloat();
	val.y = root.get(Json::ArrayIndex(1), 0).asFloat();
	val.z = root.get(Json::ArrayIndex(2), 0).asFloat();
	val.w = root.get(Json::ArrayIndex(3), 0).asFloat();
	return val;
}

Json::Value JsonSerializer::serializeMat3(glm::mat3 val)
{
	Json::Value root;
	root[0] = serializeVec3(val[0]);
	root[1] = serializeVec3(val[1]);
	root[2] = serializeVec3(val[2]);
	return root;
}

glm::mat3 JsonSerializer::deserializeMat3(Json::Value& root)
{
	glm::mat3 val;
	val[0] = deserializeVec3(root[0]);
	val[1] = deserializeVec3(root[1]);
	val[2] = deserializeVec3(root[2]);
	return val;
}

Json::Value JsonSerializer::serializeMat4(glm::mat4 val)
{
	Json::Value root;
	root[0] = serializeVec4(val[0]);
	root[1] = serializeVec4(val[1]);
	root[2] = serializeVec4(val[2]);
	root[3] = serializeVec4(val[3]);
	return root;
}

glm::mat4 JsonSerializer::deserializeMat4(Json::Value& root)
{
	glm::mat4 val;
	val[0] = deserializeVec4(root[0]);
	val[1] = deserializeVec4(root[1]);
	val[2] = deserializeVec4(root[2]);
	val[3] = deserializeVec4(root[3]);
	return val;
}
