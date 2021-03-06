#include "JsonSerializer.h"
#include "Component.h"

#include <json/reader.h>
#include <json/writer.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include "ReflectionUtils.h"

namespace spark
{
bool JsonSerializer::writeToFile(const std::filesystem::path& filePath, Json::Value& root)
{
    try
    {
        Json::StreamWriterBuilder builder;
        std::ofstream file(filePath);
        Json::StreamWriter* writer = builder.newStreamWriter();
        writer->write(root, &file);
        return true;
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
        return false;
    }
}

Json::Value JsonSerializer::readFromFile(const std::filesystem::path& filePath)
{
    Json::Value root;
    std::ifstream file(filePath, std::ios::in | std::ios::binary | std::ios::ate);
    if(file.is_open())
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

JsonSerializer* JsonSerializer::getInstance()
{
    static JsonSerializer serializer;
    return &serializer;
}

bool JsonSerializer::isPtr(const rttr::type& type)
{
    return type.is_pointer() || isWrappedPtr(type);
}

bool JsonSerializer::isWrappedPtr(const rttr::type& type)
{
    return type.is_wrapper() && type.get_wrapped_type().is_pointer();
}

bool JsonSerializer::areVariantsEqualPointers(const rttr::variant& var1, const rttr::variant& var2)
{
    const rttr::type type1{var1.get_type()}, type2{var2.get_type()};
    if(isPtr(type1) && isPtr(type2))
    {
        const rttr::variant first{isWrappedPtr(type1) ? var1.extract_wrapped_value() : var1},
            second{isWrappedPtr(type2) ? var2.extract_wrapped_value() : var2};
        return getPtr(first) == getPtr(second);
    }
    return false;
}

void* JsonSerializer::getPtr(const rttr::variant& var)
{
    const rttr::variant& resultVar = isWrappedPtr(var.get_type()) ? var.extract_wrapped_value() : var;
    if(resultVar.is_valid())
    {
        return resultVar.get_value<void*>();
    }
    SPARK_ERROR("Invalid variant of type '{}' provided as pointer type!", var.get_type().get_name().cbegin());
    throw std::runtime_error("Invalid variant provided as pointer type!");
}

std::shared_ptr<Scene> JsonSerializer::loadSceneFromFile(const std::filesystem::path& filePath)
{
    Json::Value root = readFromFile(filePath);
    return loadJsonShared<Scene>(root);
}

bool JsonSerializer::saveSceneToFile(const std::shared_ptr<Scene>& scene, const std::filesystem::path& filePath)
{
    Json::Value root;
    bool saved = save(scene, root);
    if(!saved)
    {
        return false;
    }
    return writeToFile(filePath, root);
}

bool JsonSerializer::save(const rttr::variant& var, Json::Value& root)
{
    std::lock_guard lock(serializerMutex);
    counter = 0;
    try
    {
        serialize(var, root);
        bindings.clear();
        return true;
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
        bindings.clear();
        return false;
    }
}

bool JsonSerializer::save(const rttr::variant& var, const std::filesystem::path& filePath)
{
    Json::Value root;
    bool success = save(var, root);
    if(!success)
    {
        return false;
    }
    return writeToFile(filePath, root);
}

rttr::variant JsonSerializer::loadVariant(const Json::Value& root)
{
    std::lock_guard lock(serializerMutex);
    counter = 0;
    try
    {
        rttr::variant returnValue = deserialize(root);
        bindings.clear();
        return returnValue;
    }
    catch(std::exception& e)
    {
        SPARK_ERROR("{}", e.what());
        bindings.clear();
        return nullptr;
    }
}

rttr::variant JsonSerializer::loadVariant(const std::filesystem::path& filePath)
{
    Json::Value root{readFromFile(filePath)};
    return loadVariant(root);
}

rttr::variant JsonSerializer::tryConvertVar(rttr::variant& variant, const rttr::type& type, bool& conversionOk)
{
    conversionOk = true;
    SPARK_TRACE("Converting variant of type '{}' to type '{}'...", variant.get_type().get_name().cbegin(), type.get_name().cbegin());
    if(type == variant.get_type())
    {
        SPARK_TRACE("Already the same type. Returning...");
        return variant;
    }
    if(variant.can_convert(type))
    {
        SPARK_TRACE("Using custom converter...");
        rttr::variant newVar = variant;
        bool convOk = newVar.convert(type);
        if(convOk)
        {
            return newVar;
        }
        SPARK_WARN("Failed to use custom converter!");
    }
    if(variant.get_type() == rttr::type::get(nullptr) && type.is_pointer())
    {
        SPARK_TRACE("Target type is a pointer and variant is a nullptr");
        return type.create();
    }
    if(isWrappedPtr(variant.get_type()) && !isWrappedPtr(type))
    {
        SPARK_TRACE("Using unwrapped pointer...");
        return variant.extract_wrapped_value();
    }
    if(isWrappedPtr(variant.get_type()) && isWrappedPtr(type))
    {
        SPARK_TRACE("Using custom wrapped pointer conversion...");
        // this is a temporary solution (or so I hope).
        // basically using a custom converter to upcast shared pointers of classes deriving from Component
        // recreates the shared pointers from raw pointers. Guess it's a RTTR-related problem.
        rttr::type wrapped{variant.get_type().get_wrapped_type()};
        if(rttr::type::get<std::shared_ptr<Component>>() == type && wrapped.is_derived_from(rttr::type::get<Component*>()))
        {
            rttr::method convMethod{wrapped.get_method("getComponentPtr")};
            rttr::variant wrappedVal{variant.extract_wrapped_value()};
            if(convMethod.is_valid() && convMethod.get_return_type() == type)
            {
                return convMethod.invoke(wrappedVal);
            }
        }
        else
        {
            SPARK_WARN("Custom wrapped pointer conversion unavailable for given type.");
        }
    }
    SPARK_WARN("Failed to convert variant of type '{}' to type '{}'.", variant.get_type().get_name().cbegin(), type.get_name().cbegin());
    conversionOk = false;
    return nullptr;
}

void JsonSerializer::writePropertyToJson(Json::Value& root, const rttr::type& type, const rttr::variant& var)
{
    SPARK_TRACE("Writing prop of type '{}'...", type.get_name().cbegin());
    if(isPtr(type))
    {
        SPARK_TRACE("Prop is a pointer type!");
        serialize(var, root);
    }
    else
    {
        SPARK_TRACE("Prop is statically allocated");
        unsigned char status = 0;  // outcome here as well!
        if(type.is_enumeration())
        {
            SPARK_TRACE("Prop is an enum");
            root = std::string(type.get_enumeration().value_to_name(var));
        }
        else if(type.is_arithmetic())
        {
            SPARK_TRACE("Prop is arithmetic");
            if(type == rttr::type::get<uint8_t>())
            {
                root = var.get_value<uint8_t>();
            }
            else if(type == rttr::type::get<uint16_t>())
            {
                root = var.get_value<uint16_t>();
            }
            else if(type == rttr::type::get<uint32_t>())
            {
                root = var.get_value<uint32_t>();
            }
            else if(type == rttr::type::get<uint64_t>())
            {
                root = var.get_value<uint64_t>();
            }
            else if(type == rttr::type::get<int8_t>())
            {
                root = var.get_value<int8_t>();
            }
            else if(type == rttr::type::get<int16_t>())
            {
                root = var.get_value<int16_t>();
            }
            else if(type == rttr::type::get<int32_t>())
            {
                root = var.get_value<int32_t>();
            }
            else if(type == rttr::type::get<int64_t>())
            {
                root = var.get_value<int64_t>();
            }
            else if(type == rttr::type::get<bool>())
            {
                root = var.get_value<bool>();
            }
            else if(type == rttr::type::get<float>())
            {
                root = var.get_value<float>();
            }
            else if(type == rttr::type::get<double>())
            {
                root = var.get_value<double>();
            }
            else
            {
                SPARK_WARN("Arithmetic prop was not recognized!");
                status = 1;
            }
        }
        else if(type.is_sequential_container())
        {
            SPARK_TRACE("Prop is a sequential container");
            rttr::variant_sequential_view seq{var.create_sequential_view()};
            for(int i = 0; i < static_cast<int>(seq.get_size()); ++i)
            {
                SPARK_TRACE("Writing prop[{}]...", i);
                writePropertyToJson(root[i], seq.get_value_type(), seq.get_value(i).extract_wrapped_value());
            }
            SPARK_TRACE("Sequential container closed");
        }
        else if(type.is_associative_container())
        {
            SPARK_TRACE("Prop is an associative container");
            rttr::variant_associative_view view{var.create_associative_view()};
            int counter = 0;
            if(view.is_key_only_type())
            {
                SPARK_TRACE("Container is a key only type");
                for(auto& item : view)
                {
                    writePropertyToJson(root[counter++], view.get_key_type(), item.first.extract_wrapped_value());
                }
            }
            else
            {
                SPARK_TRACE("Container is a key-value type");
                for(auto& item : view)
                {
                    writePropertyToJson(root[counter][0], view.get_key_type(), item.first.extract_wrapped_value());
                    writePropertyToJson(root[counter][1], view.get_value_type(), item.second.extract_wrapped_value());
                    counter++;
                }
            }
            SPARK_TRACE("Associative container closed");
        }
        else
        {
            SPARK_TRACE("Prop is a custom object");
            if(type == rttr::type::get<glm::vec2>())
            {
                const glm::vec2 vec{var.get_value<glm::vec2>()};
                for(int i = 0; i < 2; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::vec3>())
            {
                const glm::vec3 vec{var.get_value<glm::vec3>()};
                for(int i = 0; i < 3; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::vec4>())
            {
                const glm::vec4 vec{var.get_value<glm::vec4>()};
                for(int i = 0; i < 4; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::ivec2>())
            {
                const glm::ivec2 vec{var.get_value<glm::ivec2>()};
                for(int i = 0; i < 2; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::ivec3>())
            {
                const glm::ivec3 vec{var.get_value<glm::ivec3>()};
                for(int i = 0; i < 3; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::ivec4>())
            {
                const glm::ivec4 vec{var.get_value<glm::ivec4>()};
                for(int i = 0; i < 4; i++)
                {
                    root[i] = vec[i];
                }
            }
            else if(type == rttr::type::get<glm::mat2>())
            {
                const glm::mat2 mat{var.get_value<glm::mat2>()};
                for(int i = 0; i < 2; i++)
                {
                    for(int j = 0; j < 2; j++)
                    {
                        root[i][j] = mat[i][j];
                    }
                }
            }
            else if(type == rttr::type::get<glm::mat3>())
            {
                const glm::mat3 mat{var.get_value<glm::mat3>()};
                for(int i = 0; i < 3; i++)
                {
                    for(int j = 0; j < 3; j++)
                    {
                        root[i][j] = mat[i][j];
                    }
                }
            }
            else if(type == rttr::type::get<glm::mat4>())
            {
                const glm::mat4 mat{var.get_value<glm::mat4>()};
                for(int i = 0; i < 4; i++)
                {
                    for(int j = 0; j < 4; j++)
                    {
                        root[i][j] = mat[i][j];
                    }
                }
            }
            else if(type == rttr::type::get<std::string>())
            {
                root = var.get_value<std::string>();
            }
            else
            {  // expecting an statically allocated object of a non-simple type
                SPARK_TRACE("Unrecognized statically allocated object, expecting a custom non-simple type");
                serialize(var, root);
            }
        }
        switch(status)
        {
            case 1:
                SPARK_ERROR("Unknown property type: '{}'.", type.get_name().cbegin());
                throw std::runtime_error("Unknown property type!");
        }
    }
}

void JsonSerializer::serialize(const rttr::variant& var, Json::Value& root)
{
    /* if (!isPtr(var.get_type())) {
         SPARK_ERROR("Source object's type '{}' must be a pointer!", var.get_type().get_name().cbegin());
         throw std::runtime_error("Source object must be a pointer!");
     }*/
    SPARK_TRACE("Serializing var of type '{}'...", var.get_type().get_name().cbegin());
    if(isPtr(var.get_type()))
    {
        SPARK_TRACE("Var is a pointer type");
        if(isVarBound(var))
        {
            root[ID_NAME] = getBoundId(var);
            SPARK_TRACE("Var is bound to ID {}, skipping", root[ID_NAME].asInt());
        }
        else
        {
            const int id{counter++};
            SPARK_TRACE("Assigning ID {}", id);
            root[ID_NAME] = id;
            bindObject(var, id);
            // TODO: find a better way to get derived type maybe?
            rttr::variant wrapped{isWrappedPtr(var.get_type()) ? var.extract_wrapped_value() : var};
            rttr::instance inst{wrapped};
            const rttr::type derivedType{inst.get_derived_type()};
            const rttr::variant newVar{derivedType.create()};
            root[TYPE_NAME] = std::string(newVar.get_type().get_name());
            SPARK_TRACE("Is var a wrapped pointer? {}", isWrappedPtr(var.get_type()));
            Json::Value& content = root[CONTENT_NAME];
            for(rttr::property prop : derivedType.get_properties())
            {
                const rttr::variant serializableMeta{prop.get_metadata(SerializerMeta::Serializable)};
                if(!(serializableMeta.is_valid() && serializableMeta.can_convert<bool>() && !serializableMeta.get_value<bool>()))
                {
                    Json::Value& obj{content[std::string(prop.get_name())]};
                    SPARK_TRACE("Writing property with name '{}'...", prop.get_name().cbegin());
                    writePropertyToJson(obj, prop.get_type(), prop.get_value(wrapped));
                }
                else
                {
                    SPARK_TRACE("Skipping unserializable property with name '{}'", prop.get_name().cbegin());
                }
            }
        }
    }
    else
    {
        SPARK_TRACE("Var is not a pointer type");
        root[TYPE_NAME] = std::string(rttr::instance(var).get_derived_type().get_name());
        Json::Value& content = root[CONTENT_NAME];
        for(rttr::property prop : var.get_type().get_properties())
        {
            Json::Value& obj{content[std::string(prop.get_name())]};
            writePropertyToJson(obj, prop.get_type(), prop.get_value(var));
        }
    }
}

// todo: add outcome in read and write
rttr::variant JsonSerializer::readPropertyFromJson(const Json::Value& root, const rttr::type& type, rttr::variant& currentValue, bool& ok)
{
    SPARK_TRACE("Reading value of type '{}'...", type.get_name().cbegin());
    ok = true;
    unsigned char status = 0;  // 0 = success. Waiting for outcome here!
    if(isPtr(type))
    {
        SPARK_TRACE("Value is a pointer");
        rttr::variant sparkPtr = deserialize(root);
        // todo: i'd really love to see outcome library arrive here as well xd
        bool conversionOk;
        rttr::variant result{tryConvertVar(sparkPtr, type, conversionOk)};
        if(conversionOk)
        {
            return result;
        }
        status = 3;
        SPARK_WARN("Property of type '{}' could not be properly deserialized with var of type '{}' (valid: {})!", type.get_name().cbegin(),
                   sparkPtr.get_type().get_name().cbegin(), sparkPtr.is_valid());
    }
    else if(root.isObject() && root.isMember(TYPE_NAME) && root.isMember(CONTENT_NAME))
    {  // statically allocated object
        SPARK_TRACE("Value is a statically allocated object");
        rttr::variant valueVar = deserialize(root);
        if(valueVar.get_type() == type)
        {
            return valueVar;
        }
        if(valueVar.can_convert(type))
        {
            return valueVar.convert(type);
        }
        status = 3;
        SPARK_WARN("Property of type '{}' could not be properly deserialized with var of type '{}' (valid: {})!", type.get_name().cbegin(),
                   valueVar.get_type().get_name().cbegin(), valueVar.is_valid());
    }
    else
    {
        if(type.is_enumeration())
        {
            SPARK_TRACE("Value is an enum");
            rttr::enumeration enumeration = type.get_enumeration();
            if(root.isInt())
            {  // allow input of integral values as enums
                SPARK_TRACE("Enum is an integral value");
                int intValue = root.asInt();
                auto values = enumeration.get_values();
                auto it = std::find_if(values.begin(), values.end(), [=](const rttr::variant& var) { return intValue == var.to_int(); });
                if(it != values.end())
                {
                    return *it;
                }
                SPARK_WARN("Enum value invalid!");
                status = 2;
            }
            else
            {  // string representation expected
                SPARK_TRACE("Enum is a string representation");
                std::string stringValue = root.asString();
                auto names = enumeration.get_names();
                auto it = std::find_if(names.begin(), names.end(),
                                       [&](const rttr::basic_string_view<char>& name) { return name.compare(stringValue) == 0; });
                if(it != names.end())
                {
                    return enumeration.name_to_value(*it);
                }
                SPARK_WARN("Enum value invalid!");
                status = 2;
            }
        }
        else if(type.is_arithmetic())
        {
            SPARK_TRACE("Value is arithmetic");
            if(type == rttr::type::get<uint8_t>() || type == rttr::type::get<uint16_t>() || type == rttr::type::get<uint32_t>())
            {
                if(root.isUInt())
                {
                    return root.asUInt();
                }
                status = 2;
            }
            else if(type == rttr::type::get<uint64_t>())
            {
                if(root.isUInt64())
                {
                    return root.asUInt64();
                }
                status = 2;
            }
            else if(type == rttr::type::get<int8_t>() || type == rttr::type::get<int16_t>() || type == rttr::type::get<int32_t>())
            {
                if(root.isInt())
                {
                    return root.asInt();
                }
                status = 2;
            }
            else if(type == rttr::type::get<int64_t>())
            {
                if(root.isInt64())
                {
                    return root.asInt64();
                }
                status = 2;
            }
            else if(type == rttr::type::get<bool>())
            {
                if(root.isBool())
                {
                    return root.asBool();
                }
                status = 2;
            }
            else if(type == rttr::type::get<float>())
            {
                if(root.isDouble())
                {
                    return static_cast<float>(root.asDouble());
                }
                status = 2;
            }
            else if(type == rttr::type::get<double>())
            {
                if(root.isDouble())
                {
                    return root.asDouble();
                }
                status = 2;
            }
            else
            {
                SPARK_WARN("Arithmetic value type unknown!");
                status = 1;
            }
        }
        else if(type.is_sequential_container())
        {
            SPARK_TRACE("Value is a sequential container");
            // TODO: find a better way to deserialize containers!
            // Cannot instantiate containers with RTTR for some reason (whether it's a bug or desired behaviour)
            // For now only value replacement of statically allocated containers are supported
            if(!currentValue.is_valid() || !currentValue.is_sequential_container())
            {
                SPARK_WARN("Cannot enter the sequential container!");
                status = 2;
            }
            else
            {
                rttr::variant_sequential_view view{currentValue.create_sequential_view()};
                if(view.is_dynamic())
                {
                    SPARK_TRACE("Container is dynamic");
                    for(uint32_t i = 0; i < static_cast<uint32_t>(root.size()); ++i)
                    {
                        bool isOk;
                        rttr::variant defVal{rttr::variant()};
                        rttr::variant val{readPropertyFromJson(root[i], view.get_value_type(), defVal, isOk)};
                        if(isOk)
                        {
                            view.insert(view.begin() + i, val);
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                }
                else
                {
                    SPARK_TRACE("Container is static");
                    if(root.size() != view.get_size())
                    {
                        SPARK_WARN("Sequential container size mismatch! Read {}, expected {}.", root.size(), view.get_size());
                    }
                    for(uint32_t i = 0; i < std::min(static_cast<unsigned int>(view.get_size()), root.size()); ++i)
                    {
                        bool isOk;
                        rttr::variant currVal{view.get_value(i).extract_wrapped_value()};
                        rttr::variant val{readPropertyFromJson(root[i], view.get_value_type(), currVal, isOk)};
                        if(isOk)
                        {
                            view.set_value(i, val);
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                }
                if(status == 0)
                {
                    return currentValue;
                }
            }
        }
        else if(type.is_associative_container())
        {
            SPARK_TRACE("Value is an associative container");
            if(!currentValue.is_valid() || !currentValue.is_associative_container())
            {
                SPARK_WARN("Cannot enter the associative container!");
                status = 2;
            }
            else
            {
                rttr::variant_associative_view view{currentValue.create_associative_view()};
                if(view.is_key_only_type())
                {
                    SPARK_TRACE("Container is a key only type");
                    for(uint32_t i = 0; i < static_cast<uint32_t>(root.size()); ++i)
                    {
                        SPARK_TRACE("Reading prop[{}]...", i);
                        if(root[i].size() != 1)
                        {
                            SPARK_WARN("Cannot read! Size of the node is not equal to 1 (key only)!");
                            status = 2;
                            break;
                        }
                        bool isOk;
                        rttr::variant defVal{rttr::variant()};
                        rttr::variant key{readPropertyFromJson(root[i], view.get_key_type(), defVal, isOk)};
                        if(!isOk)
                        {
                            status = 2;
                            break;
                        }
                        if(!view.insert(key).second)
                        {
                            status = 2;
                            break;
                        }
                    }
                }
                else
                {
                    SPARK_TRACE("Container is a key-value type");
                    for(uint32_t i = 0; i < static_cast<uint32_t>(root.size()); ++i)
                    {
                        SPARK_TRACE("Reading prop[{}]...", i);
                        if(root[i].size() != 2)
                        {
                            SPARK_WARN("Cannot read! Size of the node is not equal to 2 (key-value)!");
                            status = 2;
                            break;
                        }
                        bool isOk = true;
                        rttr::variant defVal{rttr::variant()};
                        rttr::variant key{readPropertyFromJson(root[i][0], view.get_key_type(), defVal, isOk)};
                        if(!isOk)
                        {
                            status = 2;
                            break;
                        }
                        defVal = rttr::variant();
                        rttr::variant value{readPropertyFromJson(root[i][1], view.get_value_type(), defVal, isOk)};
                        if(!isOk)
                        {
                            status = 2;
                            break;
                        }
                        if(!view.insert(key, value).second)
                        {
                            status = 2;
                            break;
                        }
                    }
                }
                if(status == 0)
                {
                    return currentValue;
                }
            }
        }
        else
        {
            SPARK_TRACE("Value is a custom object");
            if(type == rttr::type::get<glm::vec2>())
            {
                if(root.size() == 2)
                {
                    glm::vec2 vec;
                    for(int i = 0; i < 2; i++)
                    {
                        if(root[i].isDouble())
                        {
                            vec[i] = static_cast<float>(root[i].asDouble());
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::vec3>())
            {
                if(root.size() == 3)
                {
                    glm::vec3 vec;
                    for(int i = 0; i < 3; i++)
                    {
                        if(root[i].isDouble())
                        {
                            vec[i] = static_cast<float>(root[i].asDouble());
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::vec4>())
            {
                if(root.size() == 4)
                {
                    glm::vec4 vec;
                    for(int i = 0; i < 4; i++)
                    {
                        if(root[i].isDouble())
                        {
                            vec[i] = static_cast<float>(root[i].asDouble());
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::ivec2>())
            {
                if(root.size() == 2)
                {
                    glm::ivec2 vec;
                    for(int i = 0; i < 2; i++)
                    {
                        if(root[i].isInt())
                        {
                            vec[i] = root[i].asInt();
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::ivec3>())
            {
                if(root.size() == 3)
                {
                    glm::ivec3 vec;
                    for(int i = 0; i < 3; i++)
                    {
                        if(root[i].isInt())
                        {
                            vec[i] = root[i].asInt();
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::ivec4>())
            {
                if(root.size() == 4)
                {
                    glm::ivec4 vec;
                    for(int i = 0; i < 4; i++)
                    {
                        if(root[i].isInt())
                        {
                            vec[i] = root[i].asInt();
                        }
                        else
                        {
                            status = 2;
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return vec;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::mat2>())
            {
                if(root.size() == 2)
                {
                    glm::mat2 mat;
                    for(int i = 0; i < 2; i++)
                    {
                        if(root[i].size() == 2)
                        {
                            for(int j = 0; j < 2; j++)
                            {
                                if(root[i][j].isDouble())
                                {
                                    mat[i][j] = static_cast<float>(root[i][j].asDouble());
                                }
                                else
                                {
                                    status = 2;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            status = 2;
                        }
                        if(status != 0)
                        {
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return mat;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::mat3>())
            {
                if(root.size() == 3)
                {
                    glm::mat3 mat;
                    for(int i = 0; i < 3; i++)
                    {
                        if(root[i].size() == 3)
                        {
                            for(int j = 0; j < 3; j++)
                            {
                                if(root[i][j].isDouble())
                                {
                                    mat[i][j] = static_cast<float>(root[i][j].asDouble());
                                }
                                else
                                {
                                    status = 2;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            status = 2;
                        }
                        if(status != 0)
                        {
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return mat;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<glm::mat4>())
            {
                if(root.size() == 4)
                {
                    glm::mat4 mat;
                    for(int i = 0; i < 4; i++)
                    {
                        if(root[i].size() == 4)
                        {
                            for(int j = 0; j < 4; j++)
                            {
                                if(root[i][j].isDouble())
                                {
                                    mat[i][j] = static_cast<float>(root[i][j].asDouble());
                                }
                                else
                                {
                                    status = 2;
                                    break;
                                }
                            }
                        }
                        else
                        {
                            status = 2;
                        }
                        if(status != 0)
                        {
                            break;
                        }
                    }
                    if(status == 0)
                    {
                        return mat;
                    }
                }
                else
                {
                    status = 2;
                }
            }
            else if(type == rttr::type::get<std::string>())
            {
                std::string string{root.asString()};
                return string;
            }
            else
            {
                SPARK_WARN("Custom object type unknown!");
                status = 1;
            }
        }
    }
    switch(status)
    {
        default:
            SPARK_ERROR("Unexpected deserialization failure!");
            throw std::runtime_error("Unexpected deserialization failure!");
        case 1:
            SPARK_ERROR("Unknown property type: '{}'.", type.get_name().cbegin());
            throw std::runtime_error("Unknown property type!");
        case 2:
            SPARK_WARN("Invalid json value given for type '{}'! Zero value will be used.", type.get_name().cbegin());
            ok = false;
            return 0;
        case 3:
            SPARK_ERROR("Invalid conversion attempt!");
            throw std::runtime_error("Invalid conversion attempt!");
    }
}

rttr::variant JsonSerializer::deserialize(const Json::Value& root)
{
    // if (!root.isMember(ID_NAME)) {
    //    SPARK_ERROR("Invalid serialization object found!");
    //    throw std::runtime_error("Invalid serialization object found!"); //maybe outcome in the future?
    //}
    SPARK_TRACE("Deserializing...");
    if(root.isMember(ID_NAME))
    {  // referenced pointer type
        SPARK_TRACE("Node is a pointer");
        const int id{root[ID_NAME].asInt()};
        if(id == NULL_ID)
        {
            SPARK_TRACE("Node is null");
            return nullptr;
        }
        if(isIdBound(id))
        {
            SPARK_TRACE("Node is bound to {}, returning", id);
            return getBoundObject(id);
        }
        if(!root.isMember(TYPE_NAME) || !root.isMember(CONTENT_NAME))
        {
            SPARK_ERROR("No type/content info provided!");
            throw std::runtime_error("No type/content info provided!");
        }
        std::string typeName{root[TYPE_NAME].asString()};
        rttr::type type{rttr::type::get_by_name(typeName)};
        if(!type.is_valid())
        {
            SPARK_ERROR("Invalid type found for name '{}'!", typeName);
            throw std::runtime_error("Invalid type found!");
        }
        SPARK_TRACE("Node is of type '{}'", type.get_name().cbegin());
        const Json::Value& content{root[CONTENT_NAME]};
        rttr::type targetType{(isWrappedPtr(type) ? type.get_wrapped_type() : type).get_raw_type()};
        SPARK_TRACE("Is pointer wrapped? {}", isWrappedPtr(type));
        rttr::variant var{targetType.create()};
        if(!var.is_valid())
        {
            SPARK_ERROR("Created variant of '{}' is not valid!", targetType.get_name().cbegin());
            throw std::runtime_error("Created variant is not valid!");
        }
        if(var.get_type() != type)
        {
            SPARK_ERROR("Created variant's type '{}' does not match source type '{}'!", var.get_type().get_name().cbegin(), type.get_name().begin());
            throw std::runtime_error("Created variant's type does not match source type!");
        }
        bindObject(var, id);
        rttr::variant wrapped{isWrappedPtr(type) ? var.extract_wrapped_value() : var};
        for(rttr::property prop : wrapped.get_type().get_properties())
        {
            const rttr::variant serializableMeta{prop.get_metadata(SerializerMeta::Serializable)};
            if(!(serializableMeta.is_valid() && serializableMeta.can_convert<bool>() && !serializableMeta.get_value<bool>()))
            {
                const rttr::type propType{prop.get_type()};
                if(content.isMember(prop.get_name().cbegin()))
                {
                    SPARK_TRACE("Reading prop with name '{}'...", prop.get_name().cbegin());
                    const Json::Value& obj{content[prop.get_name().cbegin()]};
                    bool ok = true;
                    rttr::variant propValue{prop.get_value(wrapped)};
                    rttr::variant propVar{readPropertyFromJson(obj, propType, propValue, ok)};
                    if(ok)
                    {
                        SPARK_TRACE("Acquired variant of type '{}'. Setting value...", propVar.get_type().get_name().cbegin());
                        if(!prop.set_value(wrapped, propVar))
                        {
                            SPARK_TRACE("Failed! Attempting to acquire converted variant...");
                            rttr::variant convVar{tryConvertVar(propVar, prop.get_type(), ok)};
                            if(ok)
                            {
                                SPARK_TRACE("Acquired converted variant of type '{}'. Setting value...", convVar.get_type().get_name().cbegin());
                            }
                            if(ok && !prop.set_value(wrapped, convVar))
                            {
                                SPARK_WARN("Unable to set value for property '{}' of type '{}' with converted value of type '{}'!",
                                           prop.get_name().cbegin(), propType.get_name().cbegin(), convVar.get_type().get_name().cbegin());
                            }
                            else
                            {
                                SPARK_WARN("Unable to set value for property '{}' of type '{}' with value of type '{}'!", prop.get_name().cbegin(),
                                           propType.get_name().cbegin(), propVar.get_type().get_name().cbegin());
                            }
                        }
                    }
                    else
                    {
                        SPARK_ERROR("Failed to read the property! Ignoring the value.");
                    }
                }
                else
                {
                    SPARK_WARN("Property '{}' of type '{}' (ID: {}) does not exist in json entry!", prop.get_name().cbegin(),
                               wrapped.get_type().get_name().cbegin(), id);
                }
            }
            else
            {
                SPARK_TRACE("Skipping unserializable property with name '{}'", prop.get_name().cbegin());
            }
        }
        return var;
    }
    else
    {  // statically allocated type
        SPARK_TRACE("Node is statically allocated");
        if(!root.isMember(TYPE_NAME) || !root.isMember(CONTENT_NAME))
        {
            SPARK_ERROR("No type/content info provided!");
            throw std::runtime_error("No type/content info provided!");
        }
        std::string typeName{root[TYPE_NAME].asString()};
        rttr::type type{rttr::type::get_by_name(typeName)};
        if(!type.is_valid())
        {
            SPARK_ERROR("Invalid type found for name '{}'!", typeName);
            throw std::runtime_error("Invalid type found!");
        }
        SPARK_TRACE("Node is of type '{}'", type.get_name().cbegin());
        const Json::Value& content{root[CONTENT_NAME]};
        rttr::variant var{type.create()};
        if(!var.is_valid())
        {
            SPARK_ERROR("Created variant of '{}' is not valid!", type.get_name().cbegin());
            throw std::runtime_error("Created variant is not valid!");
        }
        if(var.get_type() != type)
        {
            SPARK_ERROR("Created variant's type '{}' does not match source type '{}'!", var.get_type().get_name().cbegin(), type.get_name().begin());
            throw std::runtime_error("Created variant's type does not match source type!");
        }
        for(rttr::property prop : var.get_type().get_properties())
        {
            const rttr::type propType{prop.get_type()};
            if(content.isMember(prop.get_name().cbegin()))
            {
                SPARK_TRACE("Reading prop with name '{}'...", prop.get_name().cbegin());
                const Json::Value& obj{content[prop.get_name().cbegin()]};
                bool ok = true;
                rttr::variant propValue{prop.get_value(var)};
                rttr::variant propVar{readPropertyFromJson(obj, propType, propValue, ok)};
                if(ok && !prop.set_value(var, propVar))
                {
                    SPARK_ERROR("Unable to set value for property '{}'!", prop.get_name().cbegin());
                    throw std::runtime_error("Unable to set value for property!");
                }
            }
            else
            {
                SPARK_WARN("Property '{}' of type '{}' does not exist in json entry!", prop.get_name().cbegin(), var.get_type().get_name().cbegin());
            }
        }
        return var;
    }
}

bool JsonSerializer::bindObject(const rttr::variant& var, int id)
{
    if(isIdBound(id))
    {
        SPARK_ERROR("Provided ID is already bound!");
        throw std::runtime_error("Provided ID is already bound!");
    }
    if(isVarBound(var))
    {
        SPARK_ERROR("Provided variant is already bound!");
        throw std::runtime_error("Provided variant is already bound!");
    }
    bindings.emplace_back(var, id);
    return true;
}

bool JsonSerializer::isVarBound(const rttr::variant& var)
{
    if(getPtr(var) == nullptr)
    {
        return true;
    }
    const auto it = std::find_if(bindings.begin(), bindings.end(),
                                 [&](const std::pair<rttr::variant, int>& pair) { return areVariantsEqualPointers(pair.first, var); });
    return it != bindings.end();
}

int JsonSerializer::getBoundId(const rttr::variant& var)
{
    if(getPtr(var) == nullptr)
    {
        return -1;
    }
    const auto it = std::find_if(bindings.begin(), bindings.end(),
                                 [&](const std::pair<rttr::variant, int>& pair) { return areVariantsEqualPointers(pair.first, var); });
    if(it != bindings.end())
    {
        return it->second;
    }
    SPARK_ERROR("Unbound objects do not have an identifier!");
    throw std::runtime_error("Unbound objects do not have an identifier!");
}

bool JsonSerializer::isIdBound(const int id)
{
    const auto it = std::find_if(bindings.begin(), bindings.end(), [=](const std::pair<rttr::variant, int>& pair) { return pair.second == id; });
    return it != bindings.end();
}

rttr::variant JsonSerializer::getBoundObject(const int id)
{
    const auto it = std::find_if(bindings.begin(), bindings.end(), [=](const std::pair<rttr::variant, int>& pair) { return pair.second == id; });
    if(it != bindings.end())
    {
        return it->first;
    }
    SPARK_ERROR("Unknown identifier provided!");
    throw std::runtime_error("Unknown identifier provided!");
}
}  // namespace spark
