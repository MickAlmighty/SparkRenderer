#include "pch.h"

#include "Component.h"
#include <rttr/registration>
#include "Camera.h"

TEST(ReflectionTest, ComponentReflectedProperly) {
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

TEST(ReflectionTest, CameraPropertiesReflectedProperly) {
	spark::Camera camera{};
}