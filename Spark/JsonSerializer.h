#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include <rttr/registration>
#include <json/value.h>
#include <glm/glm.hpp>
#include <mutex>
#include <regex>
#include "Logging.h"

namespace std {
    namespace filesystem {
        class path;
    }
}

namespace spark {
    class Scene;

    class JsonSerializer final {
    public:
        ~JsonSerializer() = default;
        JsonSerializer(JsonSerializer&) = delete;
        JsonSerializer(JsonSerializer&&) = delete;
        JsonSerializer& operator+(const JsonSerializer&) = delete;
        JsonSerializer& operator+(JsonSerializer&&) = delete;
        static bool writeToFile(const std::filesystem::path& filePath, Json::Value & root);
        static Json::Value readFromFile(const std::filesystem::path & filePath);
        static JsonSerializer* getInstance();
        static bool isPtr(const rttr::type& type);
        static bool isWrappedPtr(const rttr::type & type);
        static bool areVariantsEqualPointers(const rttr::variant& var1, const rttr::variant& var2);
        static void* getPtr(const rttr::variant& var);
        std::shared_ptr<Scene> loadSceneFromFile(const std::filesystem::path& filePath);
        bool saveSceneToFile(const std::shared_ptr<Scene>& scene, const std::filesystem::path& filePath);
        bool save(const rttr::variant& var, Json::Value& root);
        bool save(const rttr::variant& var, const std::filesystem::path& filePath);
        template <typename T>
        std::shared_ptr<T> loadJson(const Json::Value& root);
        template <typename T>
        std::shared_ptr<T> load(const std::filesystem::path& filePath);
        rttr::variant loadVariant(const Json::Value& root);
        rttr::variant loadVariant(const std::filesystem::path& filePath);
    private:
        void writePropertyToJson(Json::Value& root, const rttr::type& type, const rttr::variant& var);
        rttr::variant readPropertyFromJson(const Json::Value& root, const rttr::type& type, rttr::variant& currentValue, bool& ok);
        void serialize(const rttr::variant& var, Json::Value& root);
        rttr::variant deserialize(const Json::Value& root);
        JsonSerializer() = default;
        bool bindObject(const rttr::variant& var, int id);
        bool isVarBound(const rttr::variant & var);
        int getBoundId(const rttr::variant& var);
        bool isIdBound(int id);
        rttr::variant getBoundObject(int id);
        int counter{};
        std::vector<std::pair<rttr::variant, int>> bindings;
        std::mutex serializerMutex;
        const std::string ID_NAME{ "Identifier" }, TYPE_NAME{ "Type" }, CONTENT_NAME{ "Content" };
        const int NULL_ID = -1;
    };

    template <typename T>
    std::shared_ptr<T> JsonSerializer::loadJson(const Json::Value& root) {
        std::lock_guard lock(serializerMutex);
        counter = 0;
        try {
            rttr::variant var{ deserialize(root) };
            bindings.clear();
            if(var.is_type<std::shared_ptr<T>>()) {
                return var.get_value<std::shared_ptr<T>>();
            }
            return nullptr;
        } catch(std::exception& e) {
            SPARK_ERROR("{}", e.what());
            bindings.clear();
            return nullptr;
        }
    }

    template <typename T>
    std::shared_ptr<T> JsonSerializer::load(const std::filesystem::path& filePath) {
        Json::Value root{ readFromFile(filePath) };
        return loadJson<T>(root);
    }
}
#endif