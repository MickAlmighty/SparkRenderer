#include "pch.h"

#include "Component.h"
#include "Camera.h"

#include <rttr/registration>
#include "Timer.h"
#include "JsonSerializer.h"

TEST(ReflectionTest, ComponentPropertiesValid) {
    rttr::type type{ rttr::type::get<spark::Component>() };
    ASSERT_EQ(3, type.get_properties().size());
    ASSERT_STREQ("Component", type.get_name().cbegin());
    rttr::property activeProp {type.get_property("active")}, nameProp{ type.get_property("name") },
        gameObjectProp{ type.get_property("gameObject") };
    ASSERT_TRUE(activeProp.is_valid());
    ASSERT_TRUE(nameProp.is_valid());
    ASSERT_TRUE(gameObjectProp.is_valid());
    ASSERT_STREQ("bool", activeProp.get_type().get_name().cbegin());
}

TEST(ReflectionTest, CameraReflectedAndInheritsComponent) {
    rttr::type cameraType{ rttr::type::get<spark::Camera>() }, componentType{ rttr::type::get<spark::Component>() };
    ASSERT_TRUE(cameraType.is_valid());
    ASSERT_TRUE(cameraType == rttr::type::get_by_name("Camera"));
    ASSERT_EQ(1, cameraType.get_base_classes().size());
    ASSERT_STREQ("Component", cameraType.get_base_classes().begin()->get_name().cbegin());
    ASSERT_FALSE(componentType.get_derived_classes().empty());
}
TEST(ReflectionTest, CameraValidAndAccessible) {
    rttr::type type{ rttr::type::get<spark::Camera>() };
    rttr::variant variant{ type.create() };
    ASSERT_TRUE(variant.is_valid());
    ASSERT_TRUE(variant.is_type<std::shared_ptr<spark::Camera>>());
    std::shared_ptr<spark::Camera> camera{ type.create().get_value<std::shared_ptr<spark::Camera>>() };
    ASSERT_EQ(14 + rttr::type::get<spark::Component>().get_properties().size(), type.get_properties().size());
    ASSERT_TRUE(type.get_property("cameraTarget").is_valid());
    ASSERT_TRUE(type.get_property("Position").is_valid());
    ASSERT_TRUE(type.get_property("Front").is_valid());
    ASSERT_TRUE(type.get_property("Up").is_valid());
    ASSERT_TRUE(type.get_property("Right").is_valid());
    ASSERT_TRUE(type.get_property("Yaw").is_valid());
    ASSERT_TRUE(type.get_property("Pitch").is_valid());
    ASSERT_TRUE(type.get_property("MovementSpeed").is_valid());
    ASSERT_TRUE(type.get_property("MouseSensitivity").is_valid());
    ASSERT_TRUE(type.get_property("Zoom").is_valid());
    ASSERT_TRUE(type.get_property("cameraMode").is_valid());
    ASSERT_TRUE(type.get_property("fov").is_valid());
    ASSERT_TRUE(type.get_property("zNear").is_valid());
    ASSERT_TRUE(type.get_property("zFar").is_valid());
    ASSERT_EQ(camera->getCameraTarget(), type.get_property_value("cameraTarget", camera).get_value<glm::vec3>());
    ASSERT_EQ(camera->getPosition(), type.get_property_value("Position", camera).get_value<glm::vec3>());
    ASSERT_EQ(camera->getFront(), type.get_property_value("Front", camera).get_value<glm::vec3>());
    ASSERT_EQ(camera->getUp(), type.get_property_value("Up", camera).get_value<glm::vec3>());
    ASSERT_EQ(camera->getRight(), type.get_property_value("Right", camera).get_value<glm::vec3>());
    ASSERT_EQ(camera->getYaw(), type.get_property_value("Yaw", camera).get_value<float>());
    ASSERT_EQ(camera->getPitch(), type.get_property_value("Pitch", camera).get_value<float>());
    ASSERT_EQ(camera->getMovementSpeed(), type.get_property_value("MovementSpeed", camera).get_value<float>());
    ASSERT_EQ(camera->getMouseSensitivity(), type.get_property_value("MouseSensitivity", camera).get_value<float>());
    ASSERT_EQ(camera->getZoom(), type.get_property_value("Zoom", camera).get_value<float>());
    ASSERT_EQ(camera->getCameraMode(), type.get_property_value("cameraMode", camera).get_value<spark::CameraMode>());
    ASSERT_EQ(camera->getFov(), type.get_property_value("fov", camera).get_value<float>());
    ASSERT_EQ(camera->getNearPlane(), type.get_property_value("zNear", camera).get_value<float>());
    ASSERT_EQ(camera->getFarPlane(), type.get_property_value("zFar", camera).get_value<float>());
}

TEST(ReflectionTest, PointerTypesRecognizable) {
    rttr::type falseType{ rttr::type::get<void>() };
    rttr::type sharedCamType{ rttr::type::get<std::shared_ptr<spark::Camera>>() },
        weakCamType{ rttr::type::get<std::weak_ptr<spark::Camera>>() },
        rawCamType{ rttr::type::get<spark::Camera*>() };

    using serializer = spark::JsonSerializer;
    ASSERT_FALSE(serializer::isPtr(falseType));
    ASSERT_FALSE(serializer::isWrappedPtr(falseType));
    ASSERT_TRUE(serializer::isPtr(sharedCamType));
    ASSERT_TRUE(serializer::isWrappedPtr(sharedCamType));
    ASSERT_TRUE(serializer::isPtr(weakCamType));
    ASSERT_TRUE(serializer::isWrappedPtr(weakCamType));
    ASSERT_TRUE(serializer::isPtr(rawCamType));
    ASSERT_FALSE(serializer::isWrappedPtr(rawCamType));
}

TEST(ReflectionTest, PointersComparable) {
    int falseVal = 0;
    std::shared_ptr<spark::Camera> cam = std::make_shared<spark::Camera>();
    spark::Camera* rawCam{ cam.get() };
    rttr::variant shared{ cam };
    rttr::variant raw{ rawCam };
    rttr::variant falseVar{ falseVal };
    ASSERT_TRUE(spark::JsonSerializer::areVariantsEqualPointers(shared, raw));
    ASSERT_TRUE(spark::JsonSerializer::areVariantsEqualPointers(raw, shared));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(falseVar, shared));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(shared, falseVar));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(falseVar, raw));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(raw, falseVar));
}

class SerializationClass1 {
public:
    float field1{ 1.0f };
    float field2{ 2.0f };
    RTTR_ENABLE();
};

class SerializationClass2 {
public:
    SerializationClass2() = default;
    SerializationClass2(const std::shared_ptr<SerializationClass1>& class1)
        : shared(class1), weak(class1), raw(class1.get()) {}
    std::shared_ptr<SerializationClass1> getWeak() const {
        return weak.lock();
    }
    void setWeak(std::shared_ptr<SerializationClass1> shared) {
        weak = shared;
    }
    int field1{ 0 };
    int field2{ 1 };
    std::shared_ptr<SerializationClass1> shared{};
    std::weak_ptr<SerializationClass1> weak{};
    SerializationClass1* raw{ nullptr };
    RTTR_ENABLE();
};

RTTR_REGISTRATION{
    rttr::registration::class_<SerializationClass1>("SerializationClass1")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    .property("field1", &SerializationClass1::field1)
    .property("field2", &SerializationClass1::field2);

    rttr::registration::class_<SerializationClass2>("SerializationClass2")
    .constructor()(rttr::policy::ctor::as_std_shared_ptr)
    .property("field1", &SerializationClass2::field1)
    .property("field2", &SerializationClass2::field2)
    .property("shared", &SerializationClass2::shared)
    .property("weak", &SerializationClass2::getWeak, &SerializationClass2::setWeak, rttr::registration::public_access)
    .property("raw", &SerializationClass2::raw);
}

TEST(SerializationTest, PointersInterchangeable) {
    std::shared_ptr<SerializationClass1> shared = std::make_shared<SerializationClass1>();
    SerializationClass1* raw = shared.get();
    rttr::variant sharedVar = shared, rawVar = raw, wrappedVar = sharedVar.extract_wrapped_value();

    ASSERT_TRUE(sharedVar.get_type().is_wrapper());
    ASSERT_FALSE(rawVar.get_type().is_wrapper());

    ASSERT_FALSE(sharedVar.get_type().is_pointer());
    ASSERT_TRUE(wrappedVar.get_type().is_pointer());
    ASSERT_TRUE(rawVar.get_type().is_pointer());

    ASSERT_TRUE(wrappedVar.get_type() == rawVar.get_type());

    ASSERT_EQ(static_cast<void*>(raw), rawVar.get_value<void*>(), wrappedVar.get_value<void*>());
}

TEST(SerializationTest, PointersSerializedProperly) {
    std::shared_ptr<SerializationClass1> class1 = std::make_shared<SerializationClass1>();
    class1->field1 = 3.0f;
    class1->field2 = 4.0f;
    std::shared_ptr<SerializationClass2> class2 = std::make_shared<SerializationClass2>(class1);
    Json::Value root;
    spark::JsonSerializer* serializer{ spark::JsonSerializer::getInstance() };
    serializer->serialize(class2, root);
    serializer->writeToFile("test.json", root);
    rttr::variant var{ serializer->deserialize(root) };
    ASSERT_TRUE(var.is_type<std::shared_ptr<SerializationClass2>>());
    std::shared_ptr<SerializationClass2> deserializedClass2{ var.get_value<std::shared_ptr<SerializationClass2>>() };
    ASSERT_EQ(class2->field1, deserializedClass2->field1);
    ASSERT_EQ(class2->field2, deserializedClass2->field2);
    ASSERT_NE(deserializedClass2->raw, nullptr);
    ASSERT_EQ(deserializedClass2->shared.get(), deserializedClass2->raw);
    ASSERT_EQ(deserializedClass2->weak.lock(), deserializedClass2->shared);
    ASSERT_EQ(class2->raw->field1, deserializedClass2->raw->field1);
    ASSERT_EQ(class2->raw->field2, deserializedClass2->raw->field2);
}