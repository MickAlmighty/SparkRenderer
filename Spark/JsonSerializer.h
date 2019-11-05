#ifndef JSON_SERIALIZER_H
#define JSON_SERIALIZER_H

#include <rttr/registration>
#include <json/value.h>
#include <glm/glm.hpp>
#include <mutex>
#include <regex>

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
        JsonSerializer& operator+(JsonSerializer&) = delete;
        JsonSerializer& operator+(JsonSerializer&&) = delete;
        static void writeToFile(const std::filesystem::path & filePath, Json::Value & root);
        static Json::Value readFromFile(const std::filesystem::path & filePath);
        static JsonSerializer* getInstance();
        static bool isPtr(const rttr::type& type);
        static bool isWrappedPtr(const rttr::type & type);
        static bool areVariantsEqualPointers(const rttr::variant& var1, const rttr::variant& var2);
        static void* getPtr(const rttr::variant& var);
        std::shared_ptr<Scene> loadSceneFromFile(const std::filesystem::path& filePath);
        bool saveSceneToFile(const std::shared_ptr<Scene>& scene, const std::filesystem::path& filePath);
        void serialize(const rttr::variant& var, Json::Value& root);
        rttr::variant deserialize(const Json::Value& root);
    private:
        JsonSerializer() = default;
        bool bindObject(const rttr::variant& var, int id);
        bool isVarBound(const rttr::variant & var);
        int getBoundId(const rttr::variant& var);
        bool isIdBound(const int id);
        rttr::variant getBoundObject(int id);
        int counter{};
        std::vector<std::pair<rttr::variant, int>> bindings;
        std::mutex serializerMutex;
        const std::string ID_NAME{ "Identifier" }, TYPE_NAME{ "Type" }, CONTENT_NAME{ "Content" };
        const int NULL_ID = -1;
    };

    template <class T>
    std::shared_ptr<T> make() { return std::make_shared<T>(); };
}
#endif