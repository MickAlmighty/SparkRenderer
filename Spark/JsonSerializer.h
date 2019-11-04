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

    class JsonSerializer {
    public:
        ~JsonSerializer() = default;
        JsonSerializer(JsonSerializer&) = delete;
        JsonSerializer(JsonSerializer&&) = delete;
        JsonSerializer& operator+(JsonSerializer&) = delete;
        JsonSerializer& operator+(JsonSerializer&&) = delete;
        static void writeToFile(const std::filesystem::path & filePath, Json::Value & root);
        static Json::Value readFromFile(const std::filesystem::path & filePath);
        static JsonSerializer* getInstance();
        static bool isSparkSharedPtr(const rttr::type& type);
        static bool isSparkWeakPtr(const rttr::type& type);
        static bool isSparkRawPtr(const rttr::type& type);
        static bool isSparkPtr(const rttr::type& type);
        std::shared_ptr<Scene> loadSceneFromFile(const std::filesystem::path& filePath);
        bool saveSceneToFile(const std::shared_ptr<Scene>& scene, const std::filesystem::path& filePath);
    private:
        JsonSerializer() = default;
        static bool isSparkClassMatched(const rttr::type& type, const std::string& prefix, const std::string& postfix, const std::regex& regex);
        void serialize(rttr::variant, Json::Value& root);
        rttr::variant deserialize(const Json::Value& root);
        int getBoundId(const rttr::variant& var, bool createIfNonexistent = true);
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