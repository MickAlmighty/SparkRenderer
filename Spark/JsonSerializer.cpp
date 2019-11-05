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
#include "Logging.h"

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

    bool JsonSerializer::isPtr(const rttr::type& type) {
        return type.is_pointer() || isWrappedPtr(type);
    }

    bool JsonSerializer::isWrappedPtr(const rttr::type& type) {
        return type.is_wrapper() && type.get_wrapped_type().is_pointer();
    }

    bool JsonSerializer::areVariantsEqualPointers(const rttr::variant& var1, const rttr::variant& var2) {
        const rttr::type type1{ var1.get_type() }, type2{ var2.get_type() };
        if (isPtr(type1) && isPtr(type2)) {
            const rttr::variant first{ isWrappedPtr(type1) ? var1.extract_wrapped_value() : var1 },
                second{ isWrappedPtr(type2) ? var2.extract_wrapped_value() : var2 };
            return first.get_value<void*>() == second.get_value<void*>();
        }
        return false;
    }

    void* JsonSerializer::getPtr(const rttr::variant& var) {
        const rttr::variant& resultVar = isWrappedPtr(var.get_type()) ? var.extract_wrapped_value() : var;
        if(resultVar.is_valid()) {
            return resultVar.get_value<void*>();
        }
        SPARK_ERROR("Invalid variant of type '{}' provided as pointer type!", var.get_type().get_name().cbegin());
        throw std::exception("Invalid variant provided as pointer type!");
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

    void JsonSerializer::serialize(const rttr::variant& var, Json::Value& root) {
        if (!isPtr(var.get_type())) {
            SPARK_ERROR("Source object must be a pointer!");
            throw std::exception("Source object must be a pointer!");
        }
        if (isVarBound(var)) {
            root[ID_NAME] = getBoundId(var);
        } else {
            const int id{ counter++ };
            root[ID_NAME] = id;
            bindObject(var, id);
            root[TYPE_NAME] = std::string(var.get_type().get_name());
            rttr::variant wrapped{ isWrappedPtr(var.get_type()) ? var.extract_wrapped_value() : var };
            Json::Value& content = root[CONTENT_NAME];
            for (rttr::property prop : wrapped.get_type().get_properties()) {
                if (isPtr(prop.get_type())) {
                    serialize(prop.get_value(wrapped), content[std::string(prop.get_name())]);
                } else {

                }
            }
        }
    }

    rttr::variant JsonSerializer::deserialize(const Json::Value& root) {
        if (!root.isMember(ID_NAME)) {
            SPARK_ERROR("Invalid serialization object found!");
            throw std::exception("Invalid serialization object found!"); //maybe outcome in the future?
        }
        const int id{ root[ID_NAME].asInt() };
        if (id == NULL_ID) {
            return nullptr;
        }
        if (isIdBound(id)) {
            return getBoundObject(id);
        }
        if (!root.isMember(TYPE_NAME) || !root.isMember(CONTENT_NAME)) {
            SPARK_ERROR("No type/content info provided!");
            throw std::exception("No type/content info provided!");
        }
        rttr::type type{ rttr::type::get_by_name(root[TYPE_NAME].asString()) };
        if (!type.is_valid()) {
            SPARK_ERROR("Invalid type found for name '{}'!", root[TYPE_NAME].asString());
            throw std::exception("Invalid type found!");
        }
        const Json::Value& content{ root[CONTENT_NAME] };
        rttr::variant var{ type.create() };
        bindObject(var, id);
        rttr::variant wrapped{ isWrappedPtr(type) ? var.extract_wrapped_value() : var };
        for (rttr::property prop : wrapped.get_type().get_properties()) {
            if (content.isMember(prop.get_name().cbegin())) {
                const Json::Value& obj{ content[prop.get_name().cbegin()] };
                if (isPtr(prop.get_type())) {
                    rttr::variant sparkPtr = deserialize(obj);
                    prop.set_value(wrapped, sparkPtr.convert(prop.get_type()));
                } else {

                }
            } else {
                SPARK_WARN("Property {} of type {} does not exist in json entry (ID: {})",
                           prop.get_name().cbegin(), wrapped.get_type().get_name().cbegin(), id);
            }
        }
        return var;
    }

    bool JsonSerializer::bindObject(const rttr::variant& var, int id) {
        if (isIdBound(id)) {
            SPARK_ERROR("Provided ID is already bound!");
            throw std::exception("Provided ID is already bound!");
        }
        if (isVarBound(var)) {
            SPARK_ERROR("Provided variant is already bound!");
            throw std::exception("Provided variant is already bound!");
        }
        bindings.emplace_back(var, id);
        return true;
    }

    bool JsonSerializer::isVarBound(const rttr::variant& var) {
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [&](const std::pair<rttr::variant, int>& pair) {
            return areVariantsEqualPointers(pair.first, var);
        });
        return it != bindings.end();
    }

    int JsonSerializer::getBoundId(const rttr::variant& var) {
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [&](const std::pair<rttr::variant, int>& pair) {
            return areVariantsEqualPointers(pair.first, var);
        });
        if (it != bindings.end()) {
            return it->second;
        }
        SPARK_ERROR("Unbound objects do not have an identificator!");
        throw std::exception("Unbound objects do not have an identificator!");
    }

    bool JsonSerializer::isIdBound(const int id) {
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [=](const std::pair<rttr::variant, int>& pair) {
            return pair.second == id;
        });
        return it != bindings.end();
    }

    rttr::variant JsonSerializer::getBoundObject(const int id) {
        const auto it = std::find_if(bindings.begin(), bindings.end(),
                                     [=](const std::pair<rttr::variant, int>& pair) {
            return pair.second == id;
        });
        if (it != bindings.end()) {
            return it->first;
        }
        SPARK_ERROR("Unknown identifier provided!");
        throw std::exception("Unknown identifier provided!");
    }
}
