#include "pch.h"

#include "Component.h"
#include "Camera.h"
#include "JsonSerializer.h"

#include <rttr/registration>

TEST(ReflectionTest, ComponentPropertiesValid)
{
    const rttr::type type{rttr::type::get<spark::Component>()};
    ASSERT_EQ(3, type.get_properties().size());
    ASSERT_STREQ("Component", type.get_name().cbegin());
    const rttr::property activeProp{type.get_property("active")}, nameProp{type.get_property("name")}, gameObjectProp{type.get_property("gameObject")};
    ASSERT_TRUE(activeProp.is_valid());
    ASSERT_TRUE(nameProp.is_valid());
    ASSERT_TRUE(gameObjectProp.is_valid());
    ASSERT_STREQ("bool", activeProp.get_type().get_name().cbegin());
}

TEST(ReflectionTest, CameraReflectedAndInheritsComponent)
{
    const rttr::type cameraType{rttr::type::get<spark::Camera>()}, componentType{rttr::type::get<spark::Component>()};
    ASSERT_TRUE(cameraType.is_valid());
    ASSERT_TRUE(cameraType == rttr::type::get_by_name("Camera"));
    ASSERT_EQ(1, cameraType.get_base_classes().size());
    ASSERT_STREQ("Component", cameraType.get_base_classes().begin()->get_name().cbegin());
    ASSERT_FALSE(componentType.get_derived_classes().empty());
}
TEST(ReflectionTest, CameraValidAndAccessible)
{
    const rttr::type type{rttr::type::get<spark::Camera>()};
    const rttr::variant variant{type.create()};
    ASSERT_TRUE(variant.is_valid());
    ASSERT_TRUE(variant.is_type<std::shared_ptr<spark::Camera>>());
    const auto camera{type.create().get_value<std::shared_ptr<spark::Camera>>()};
    ASSERT_EQ(rttr::type::get<spark::Camera>().get_properties().size(), type.get_properties().size());
    ASSERT_TRUE(type.get_property("position").is_valid());
    ASSERT_TRUE(type.get_property("front").is_valid());
    ASSERT_TRUE(type.get_property("up").is_valid());
    ASSERT_TRUE(type.get_property("right").is_valid());
    ASSERT_TRUE(type.get_property("yaw").is_valid());
    ASSERT_TRUE(type.get_property("pitch").is_valid());
    ASSERT_TRUE(type.get_property("movementSpeed").is_valid());
    ASSERT_TRUE(type.get_property("mouseSensitivity").is_valid());
    ASSERT_TRUE(type.get_property("fov").is_valid());
    ASSERT_TRUE(type.get_property("zNear").is_valid());
    ASSERT_TRUE(type.get_property("zFar").is_valid());
}

TEST(ReflectionTest, PointerTypesRecognizable)
{
    const rttr::type falseType{rttr::type::get<void>()};
    const rttr::type sharedCamType{rttr::type::get<std::shared_ptr<spark::Camera>>()}, weakCamType{rttr::type::get<std::weak_ptr<spark::Camera>>()},
                     rawCamType{rttr::type::get<spark::Camera*>()};

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

TEST(ReflectionTest, PointersComparable)
{
    int falseVal = 0;
    std::shared_ptr<spark::Camera> cam = std::make_shared<spark::Camera>();
    spark::Camera* rawCam{cam.get()};
    const rttr::variant shared{cam};
    const rttr::variant raw{rawCam};
    const rttr::variant falseVar{falseVal};
    ASSERT_TRUE(spark::JsonSerializer::areVariantsEqualPointers(shared, raw));
    ASSERT_TRUE(spark::JsonSerializer::areVariantsEqualPointers(raw, shared));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(falseVar, shared));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(shared, falseVar));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(falseVar, raw));
    ASSERT_FALSE(spark::JsonSerializer::areVariantsEqualPointers(raw, falseVar));
}