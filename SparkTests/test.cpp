#include "pch.h"

#include "Component.h"
#include <rttr/registration>

TEST(ReflectionTest, ComponentReflectedProperly) {
	rttr::type type{ rttr::type::get<spark::Component>() };
	ASSERT_EQ(2, type.get_properties().size());
	ASSERT_STREQ("Component", type.get_name().cbegin());
	rttr::property activeProp {type.get_property("active")}, nameProp{ type.get_property("name") };
	ASSERT_TRUE(activeProp.is_valid());
	ASSERT_TRUE(nameProp.is_valid());
	ASSERT_STREQ("bool", activeProp.get_type().get_name().cbegin());
	ASSERT_STREQ("std::string", nameProp.get_type().get_name().cbegin());
}