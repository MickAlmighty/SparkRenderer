#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include <json/value.h>
#include <glm/glm.hpp>

namespace std {
	namespace filesystem {
		class path;
	}
}

namespace spark {

class JsonSerializer
{
public:
	static void writeToFile(std::filesystem::path&& filePath, Json::Value&& root);
	static Json::Value readFromFile(std::filesystem::path&& filePath);

	//static Json::Value serialize(const std::shared_ptr<ISerializable> objToSerialize);
	//static std::shared_ptr<ISerializable> deserialize(Json::Value& root);

	static void clearState();
	static Json::Value serializeVec2(glm::vec2 val);
	static glm::vec2 deserializeVec2(Json::Value& root);
	static Json::Value serializeVec3(glm::vec3 val);
	static glm::vec3 deserializeVec3(Json::Value& root);
	static Json::Value serializeVec4(glm::vec4 val);
	static glm::vec4 deserializeVec4(Json::Value& root);
	static Json::Value serializeMat3(glm::mat3 val);
	static glm::mat3 deserializeMat3(Json::Value& root);
	static Json::Value serializeMat4(glm::mat4 val);
	static glm::mat4 deserializeMat4(Json::Value& root);

private:
	inline static int counter = 0;
	//static std::map<std::shared_ptr<ISerializable>, int> serializedObjects;

	JsonSerializer() = default;
	~JsonSerializer() = default;

	//static std::shared_ptr<ISerializable> findSerializedObject(const int id);
	//static int findId(const std::shared_ptr<ISerializable>& serializableObject);
};

template <class T>
std::shared_ptr<T> make() { return std::make_shared<T>(); };

}
#endif