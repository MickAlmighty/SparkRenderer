#include "JsonSerializer.h"

#include <iostream>
#include <filesystem>
#include <fstream>

#include <json/reader.h>
#include <json/writer.h>

#include "ActorAI.h"
#include "Camera.h"
#include "DirectionalLight.h"
#include "GameObject.h"
#include "ISerializable.h"
#include "Mesh.h"
#include "MeshPlane.h"
#include "ModelMesh.h"
#include "PointLight.h"
#include "SpotLight.h"
#include "TerrainGenerator.h"
#include <regex>

namespace spark {

//std::map<std::shared_ptr<ISerializable>, int> JsonSerializer::serializedObjects;
//
//std::shared_ptr<ISerializable> JsonSerializer::findSerializedObject(const int id)
//{
//	for (const auto& it : serializedObjects)
//	{
//		if (it.second == id)
//		{
//			return it.first;
//		}
//	}
//	return nullptr;
//}
//
//int JsonSerializer::findId(const std::shared_ptr<ISerializable>& serializableObject)
//{
//	for (auto it : serializedObjects)
//	{
//		if (it.first == serializableObject)
//		{
//			return it.second;
//		}
//	}
//	return -1;
//}

    void JsonSerializer::writeToFile(const std::filesystem::path& filePath, Json::Value& root) {
        Json::StreamWriterBuilder builder;
        std::ofstream file(filePath);
        Json::StreamWriter* writer = builder.newStreamWriter();
        writer->write(root, &file);
    }

    Json::Value JsonSerializer::readFromFile(const std::filesystem::path& filePath) {
        Json::Value root;
        std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
        if (file.is_open()) {
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

    //Json::Value JsonSerializer::serialize(const std::shared_ptr<ISerializable> objToSerialize)
    //{
    //	Json::Value root;
    //	if(objToSerialize == nullptr)
    //	{
    //		root["id"] = -1;
    //		return root;
    //	}
    //
    //	const int id = findId(objToSerialize);
    //	if (id != -1)
    //	{
    //		//if id != -1 means that this object has been serialized already
    //		root["id"] = id;
    //		root["SerializableType"] = static_cast<int>(objToSerialize->getSerializableType());
    //		return root;
    //	}
    //
    //	counter++;
    //	serializedObjects.emplace(objToSerialize, counter);
    //	root["id"] = counter;
    //	root["SerializableType"] = static_cast<int>(objToSerialize->getSerializableType());
    //	root["object"] = objToSerialize->serialize();
    //	return root;
    //}
    //
    //std::shared_ptr<ISerializable> JsonSerializer::deserialize(Json::Value& root)
    //{
    //	int id = root.get("id", -1).asInt();
    //	if(id == -1)
    //	{
    //		return nullptr;
    //	}
    //
    //	if (const auto obj = findSerializedObject(id); obj != nullptr)
    //	{
    //		return obj;
    //	}
    //
    //	const auto type = static_cast<SerializableType>(root["SerializableType"].asInt());
    //	std::shared_ptr<ISerializable> deserialized;
    //	switch (type)
    //	{
    //	case SerializableType::SGameObject:
    //		deserialized = make<GameObject>();
    //		break;
    //	case SerializableType::SModelMesh:
    //		deserialized = make<ModelMesh>();
    //		break;
    //	case SerializableType::SMeshPlane:
    //		deserialized = make<MeshPlane>();
    //		break;
    //	case SerializableType::STerrainGenerator:
    //		deserialized = make<TerrainGenerator>();
    //		break;
    //	case SerializableType::SActorAI:
    //		deserialized = make<ActorAI>();
    //		break;
    //	case SerializableType::SCamera:
    //		deserialized = make<Camera>();
    //		break;
    //	case SerializableType::SDirectionalLight:
    //		deserialized = make<DirectionalLight>();
    //		break;
    //	case SerializableType::SPointLight:
    //		deserialized = make<PointLight>();
    //		break;
    //	case SerializableType::SSpotLight:
    //		deserialized = make<SpotLight>();
    //		break;
    //	default:
    //		throw std::exception("Unsupported SerializableType encountered!");;
    //	}
    //	serializedObjects.emplace(deserialized, id);
    //	deserialized->deserialize(root["object"]);
    //	return deserialized;
    //}

    JsonSerializer* JsonSerializer::getInstance() {
        static JsonSerializer serializer;
        return &serializer;
    }

    bool JsonSerializer::isPtr(const rttr::type & type) {
        return type.is_pointer() || type.is_wrapper();
    }

    std::shared_ptr<Scene> JsonSerializer::loadSceneFromFile(const std::filesystem::path& filePath) {
        std::lock_guard lock(serializerMutex);
        counter = 0;
        throw std::exception();
    }

    bool JsonSerializer::saveSceneToFile(const std::shared_ptr<Scene>& scene, const std::filesystem::path& filePath) {
        std::lock_guard lock(serializerMutex);
        counter = 0;
        throw std::exception();
    }

    void JsonSerializer::serialize(rttr::variant var, Json::Value& root) {
        if(!isPtr(var.get_type())) {
            throw std::exception("Source object must be a pointer!");
        }
        const int id{ counter++ };
        root[ID_NAME] = id;
        root[TYPE_NAME] = std::string(var.get_type().get_name());
        rttr::variant wrapped{ var.get_type().is_wrapper() ? var.extract_wrapped_value() : var };
        Json::Value& content = root[CONTENT_NAME];
        for(rttr::property prop : wrapped.get_type().get_properties()) {
            if(isPtr(prop.get_type())) {
                serialize(prop.get_value(wrapped), content[std::string(prop.get_name())]);
            } else {
                
            }
        }
    }

    rttr::variant JsonSerializer::deserialize(const Json::Value& root) {
        if (!root.isMember(ID_NAME)) {
            //todo: Log error (gotta add spdlog later)
            throw std::exception("Invalid serialization object found!"); //maybe outcome in the future?
        }
        const int id{ root[ID_NAME].asInt() };
        if (id == NULL_ID) {
            return nullptr;
        }
        if (rttr::variant var = getBoundObject(id); var != nullptr) {
            return var;
        }
        if (!root.isMember(TYPE_NAME) || !root.isMember(CONTENT_NAME)) {
            throw std::exception("No type/content info provided!");
        }
        rttr::type type{ rttr::type::get_by_name(root[TYPE_NAME].asString()) };
        if (!type.is_valid()) {
            throw std::exception("Invalid type found!");
        }
        const Json::Value& content{ root[CONTENT_NAME] };
        rttr::variant var{ type.create() };
        bindObject(var, id);
        rttr::variant wrapped{ type.is_wrapper() ? var.extract_wrapped_value() : var };
        for (rttr::property prop : wrapped.get_type().get_properties()) {
            if (content.isMember(prop.get_name().cbegin())) {
                const Json::Value& obj{ content[prop.get_name().cbegin()] };
                if (isPtr(prop.get_type())) {
                    rttr::variant sparkPtr = deserialize(obj);
                    prop.set_value(wrapped, sparkPtr.convert(prop.get_type()));
                } else {
                    
                }
            } else {
                //todo: warn about the property being absent
            }
        }
        return var;
    }

    bool JsonSerializer::bindObject(const rttr::variant& var, int id) {
        if (getBoundObject(id) != nullptr) {
            return false;
        }
        bindings.emplace_back(var, id);
        return true;
    }

    int JsonSerializer::getBoundId(const rttr::variant& var, bool createIfNonexistent) {
        if (var == nullptr) {
            return NULL_ID;
        }
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [&](const std::pair<rttr::variant, int>& pair) {
            const rttr::variant& first{ pair.first.get_type().is_wrapper() ? pair.first.extract_wrapped_value() : pair.first };
            const rttr::variant& second{ var.get_type().is_wrapper() ? var.extract_wrapped_value() : var };
            return first == second;
        });
        if (it != bindings.end()) {
            return it->second;
        }
        if (!createIfNonexistent) {
            throw std::exception("Unknown object encountered and createIfNonexistent is false!");
        }
        bindings.emplace_back(var, counter);
        return counter++;
    }

    rttr::variant JsonSerializer::getBoundObject(const int id) {
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [=](const std::pair<rttr::variant, int>& pair) {
            return pair.second == id;
        });
        if (it != bindings.end()) {
            return it->first;
        }
        return nullptr;
    }
}
